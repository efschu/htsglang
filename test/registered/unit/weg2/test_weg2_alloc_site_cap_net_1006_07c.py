"""nf-next-1006-07c: option C (``_cap_overask``) BUILT in the product, with the 07 matrix.

History: nf-next-1006-07 asked what is left as a NET at the allocation site when the admission
leaves a gap, as drafts on top of the base (test file a18d7048f9). The NF seat decided (06.10.
02:05Z): C is built, A0/A1/B are not. This file keeps the 07 drafts A0/A1/B as TEXT applied to the
BASE (= the product with the C hunk reverted, ``_base``), so the red-on-base proof and the replica
hazard of A1/B stay pinned; C is now the PRODUCT (``mem_cache/common.py::_cap_overask`` + the one
changed call in ``alloc_paged_token_slots_extend``) and reads the published group gap of
nf-next-1006-01 (``uniform_cap_blind_gap``, produced by the real ``publish_cap_blind_gap``).

Original 07 question (desk, CPU). cand3 (bce16a6ddf, 05.10. 23:45:03Z) died in
``alloc_paged_token_slots_extend`` (mem_cache/common.py): avail 832, 19200 evictable ABOVE the
residency cap, extend 1572; the one peel the path makes delivered 64 of 868 and the relief seam
(``_attempt_extend_relief``) is empty -> ``Prefill out of memory`` on all three D ranks. The tick
side is fixed (1540, 8acdd4fb77), the admission side is nf-next-1006-01. This file asks what a
SECOND line at the allocation site could do, and prices it:

  A0  F7 literal: call the #790 confiscation round (``_evict_past_confiscation``) on the paged path.
  A1  peel-through: a floor-compatible round (replicated ask, rank-local entry and stop).
  B   provider that peels only leaves whose ids are all <= the cap ("below first").
  C   uniform over-ask BEFORE the allocation (reads the group gap nf-next-1006-01 publishes,
      CAP_GAP_ATTR): the only variant with a rank-uniform entry. NOW THE PRODUCT.

HOW THE DRAFTS ARE TESTED. The drafts are text (``DRAFT_*`` below, identical to the diff hunks in
the report). ``_patched`` applies a draft to the SOURCE of the product function and compiles it, so a
draft that no longer fits the base fails loudly at its anchor. Pool, tree and cap are doubles
around REAL code: ``KvRowCap`` (free listener included, via ``_engage_kv_cap``),
``payable_size``, ``uniform_avail_for_evict``, ``evict_from_tree_cache``, ``publish_cap_blind_gap``
and the real ``alloc_paged_token_slots_extend`` (unpatched = the product, with C).

MODEL ASSUMPTIONS (named, not measured): the tree is a radix tree of nodes, each holding a set of
page ids; a node is a leaf when it has no child; the peel takes the oldest leaf first (the product's
LRU heap over ``evictable_device_leaves``); a freed node's parent becomes a leaf only after the
node is gone; ids are rank-local (pool sizes on NF-D differ per rank: 229447 / 294985 rows,
D.log 23:45:03 WEG2-ARENA-LOAD); the allocator hands out the lowest free ids first.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import inspect  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import types  # noqa: E402
from typing import Dict, List, Optional, Tuple  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

PAGE = 64
NUM_PAGES = 4096
CAP_TOKENS = 32768                 # cand3: cap=32768 (Z.191919)
CAP_PAGES = CAP_TOKENS // PAGE     # 512
EXTEND = 1572                      # cand3: Try to allocate 1572 tokens


# ---------------------------------------------------------------------------
# doubles
# ---------------------------------------------------------------------------
class _Pool:
    """Paged allocator double: ids 1..NUM_PAGES, free list as a tensor, the free listener
    ``KvRowCap`` subscribes to. Lowest free id first."""

    page_size = PAGE
    num_pages = NUM_PAGES

    def __init__(self, free_ids: List[int]):
        self.free_pages = torch.tensor(sorted(free_ids), dtype=torch.int64)
        self.release_pages = torch.empty(0, dtype=torch.int64)
        self.free_group: list = []
        self._on_free = []
        self._on_clear = []
        self.residency_withheld_slots = 0

    def register_free_listener(self, on_free, on_clear=None):
        self._on_free.append(on_free)
        if on_clear is not None:
            self._on_clear.append(on_clear)

    def available_size(self) -> int:
        return (int(self.free_pages.numel()) + int(self.release_pages.numel())) * PAGE

    def free_ids(self, ids: List[int]) -> None:
        self.free_pages = torch.sort(
            torch.cat((self.free_pages, torch.tensor(ids, dtype=torch.int64)))
        ).values
        for cb in self._on_free:
            cb(ids)

    def alloc_extend(self, prefix_lens, prefix_lens_cpu, seq_lens, seq_lens_cpu, last_loc,
                     extend_num_tokens, **kw):
        n = math.ceil(int(extend_num_tokens) / PAGE)
        if int(self.free_pages.numel()) < n:
            return None
        out = self.free_pages[:n].clone()
        self.free_pages = self.free_pages[n:]
        return out

    def backup_state(self):
        return None

    def flush_free_group(self):
        return 0


class _Node:
    def __init__(self, name: str, pages: List[int], parent: Optional["_Node"], age: int):
        self.name = name
        self.pages = list(pages)
        self.parent = parent
        self.children: List["_Node"] = []
        self.age = age                 # LRU: smaller = older
        if parent is not None:
            parent.children.append(self)


class _Tree:
    """Radix-tree double with real leaf structure. ``evict`` peels the oldest LEAF first and
    frees its pages to the pool (whose free listener may take them away again); a node becomes
    a leaf only when its last child is gone. ``max_page_id`` on the params (draft B) restricts
    the victims to leaves whose pages are ALL at or below that id."""

    uniform_avail_floor = None
    uniform_admitted_since_floor = 0

    def __init__(self, pool: _Pool, nodes: List[_Node]):
        self.token_to_kv_pool_allocator = pool
        self.nodes = list(nodes)
        self.evicted: List[str] = []
        self.evict_calls = 0

    # -- the surface the product reads ---------------------------------------------------------
    def is_chunk_cache(self):
        return False

    def evictable_size(self):
        return sum(len(n.pages) for n in self.nodes) * PAGE

    full_evictable_size = evictable_size

    def available_and_evictable_str(self):
        a = self.token_to_kv_pool_allocator
        return "Available full tokens: %d (full_available_size=%d + full_evictable_size_=%d)" % (
            a.available_size() + self.evictable_size(), a.available_size(), self.evictable_size())

    def pretty_print(self):
        return None

    def evict(self, params):
        self.evict_calls += 1
        want = int(params.num_tokens)
        bound = getattr(params, "max_page_id", None)
        freed = 0
        while freed < want:
            leaves = [n for n in self.nodes if not n.children]
            if bound is not None:
                leaves = [n for n in leaves if max(n.pages) <= bound]
            if not leaves:
                break
            leaf = min(leaves, key=lambda n: n.age)
            self.nodes.remove(leaf)
            if leaf.parent is not None:
                leaf.parent.children.remove(leaf)
            self.evicted.append(leaf.name)
            self.token_to_kv_pool_allocator.free_ids(list(leaf.pages))
            freed += len(leaf.pages) * PAGE
        return types.SimpleNamespace(num_tokens_evicted=freed)

    # -- the test's own view --------------------------------------------------------------------
    def above_tokens(self, cap_pages: int) -> int:
        return sum(1 for n in self.nodes for p in n.pages if p > cap_pages) * PAGE


def _ids(lo: int, hi: int) -> List[int]:
    return list(range(lo, hi + 1))


def _world(layout: Dict[str, object], free_below: List[int], floor: Optional[int] = 832,
           cap: bool = True):
    """One rank. ``layout``: name -> (parent name or None, ids, age). Returns (pool, tree)."""
    from sglang.srt.weg2 import d_seat_vram as dsv

    used = {p for (_, ids, _) in layout.values() for p in ids}
    above_free = (
        [i for i in range(CAP_PAGES + 1, NUM_PAGES + 1) if i not in used] if cap else []
    )
    pool = _Pool(list(free_below) + above_free)
    nodes: Dict[str, _Node] = {}
    for name, (parent, ids, age) in layout.items():     # parents are listed before their children
        nodes[name] = _Node(name, ids, nodes.get(parent) if parent else None, age)
    tree = _Tree(pool, list(nodes.values()))
    if cap:
        dsv._engage_kv_cap(pool, CAP_TOKENS, PAGE)       # REAL KvRowCap + free listener
    tree.uniform_avail_floor = floor                      # NF-D publishes it every iteration (#1045)
    return pool, tree


# the cand3 state of the log: avail 832 (13 pages), evictable 19200 (300 pages), delivered 64.
#   L0  1 page, below the cap, the oldest leaf       -> the 64 the peel paid
#   A   41 pages, below the cap, parent of B          -> payable, but BEHIND the leaf B
#   B   258 pages, above the cap, leaf                -> 16512 tokens that pay the pool nothing
def _cand3_rank0():
    return _world(
        {"L0": (None, [421], 1), "A": (None, _ids(422, 462), 2), "B": ("A", _ids(600, 857), 3)},
        free_below=_ids(408, 420))


# rank 1: same tree, other ids: the oldest leaf L0 sits ABOVE the cap too (first peel pays 0)
def _cand3_rank1():
    return _world(
        {"L0": (None, [1000], 1), "A": (None, _ids(422, 462), 2), "B": ("A", _ids(600, 857), 3)},
        free_below=_ids(408, 420))


# rank 2: same tree, ids all below the cap: its first peel pays in full and it never fails
def _cand3_rank2():
    return _world(
        {"L0": (None, [203], 1), "A": (None, _ids(204, 244), 2), "B": ("A", _ids(245, 502), 3)},
        free_below=_ids(190, 202))


# the chain case: an older above-cap leaf X absorbs the one peel the paged path makes; the
# payable node A sits under the above-cap leaf B and nothing wholly below the cap is a leaf
def _chain_rank0():
    return _world(
        {"X": (None, _ids(900, 939), 1), "A": (None, _ids(422, 462), 2),
         "B": ("A", _ids(600, 817), 3)},
        free_below=_ids(408, 420))


def _alloc(fn, tree: _Tree, tokens: int = EXTEND):
    t = torch.tensor([0], dtype=torch.int64)
    return fn(tree_cache=tree, prefix_lens=t, prefix_lens_cpu=t, seq_lens=t, seq_lens_cpu=t,
              last_loc=t, extend_num_tokens=tokens)


def _try(fn, tree: _Tree, tokens: int = EXTEND) -> Tuple[bool, str]:
    try:
        _alloc(fn, tree, tokens)
        return True, ""
    except RuntimeError as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# the drafts (these strings ARE the diff hunks of the report)
# ---------------------------------------------------------------------------
ANCHOR_RULE3 = (
    "    if out_cache_loc is None:\n"
    "        # #681 RULE 3: every alloc path reachable from prefill admission gets\n"
)
ANCHOR_TOP = (
    "    evict_from_tree_cache(tree_cache, num_tokens)\n"
    "    delivered = max(0, payable_size(allocator) - payable_before)\n"
    "    evict_asked = max(0, num_tokens - payable_before)\n"
)

# --- A0: F7 literal --------------------------------------------------------------------------
DRAFT_A0_HUNK = '''\
    if out_cache_loc is None:
        # nf-next-1006-07 option A0: the #790 confiscation round on the paged path (F7).
        gained = _evict_past_confiscation(tree_cache, allocator, num_tokens - delivered)
        if gained > 0:
            _flush_deferred_frees(allocator)
            if backup_state:
                state = allocator.backup_state()
            out_cache_loc = _attempt_alloc()
'''

# --- A1: peel-through, floor-compatible ------------------------------------------------------
DRAFT_A1_HELPER = '''\
def _peel_through_cap(tree_cache, allocator, base_ask: int, shortfall: int) -> int:
    """nf-next-1006-07 option A1. Peel again, PAST the leaves a residency cap confiscates.

    Only with an engaged KvRowCap (27B / P and a D without a cap never reach the loop).
    The ask is the REPLICATED num_tokens, doubled per round, never the rank-local shortfall;
    unlike _evict_past_confiscation it does not decline under uniform_avail_floor (NF-D
    publishes one every iteration, #1045). ENTRY AND STOP ARE RANK-LOCAL (delivered).
    """
    cap = getattr(allocator, "_weg2_kv_stage_cap", None)
    if cap is None or not getattr(cap, "engaged", False):
        return 0
    delivered = 0
    ask = max(1, int(base_ask))
    for _ in range(_CONFISCATION_PEEL_ROUNDS):
        before = payable_size(allocator)
        counted = int(evict_from_tree_cache(tree_cache, ask) or 0)
        delivered += max(0, payable_size(allocator) - before)
        if delivered >= shortfall or counted <= 0:
            break
        ask *= 2
    return delivered
'''
DRAFT_A1_HUNK = '''\
    if out_cache_loc is None:
        # nf-next-1006-07 option A1: peel through the confiscated leaves, then retry once.
        gained = _peel_through_cap(
            tree_cache, allocator, num_tokens,
            max(1, int(extend_num_tokens) - payable_size(allocator)),
        )
        if gained > 0:
            _flush_deferred_frees(allocator)
            if backup_state:
                state = allocator.backup_state()
            out_cache_loc = _attempt_alloc()
'''

# --- B: provider, victims restricted to leaves wholly at or below the cap -------------------------
DRAFT_B_HELPER = '''\
def _evict_wholly_below_cap(tree_cache, allocator, shortfall: int) -> int:
    """nf-next-1006-07 option B. Peel only leaves whose page ids are ALL <= the cap.

    Needs a tree-side hook: EvictParams.max_page_id (default None) threaded to
    FullComponent._peel as a victim filter (leaf.value.max() <= cap * page_size).
    """
    cap = getattr(allocator, "_weg2_kv_stage_cap", None)
    if cap is None or not getattr(cap, "engaged", False):
        return 0
    params = EvictParams(num_tokens=max(1, int(shortfall)))
    params.max_page_id = int(cap.cap)
    before = payable_size(allocator)
    tree_cache.evict(params)
    return max(0, payable_size(allocator) - before)
'''
DRAFT_B_HUNK = '''\
    if out_cache_loc is None:
        # nf-next-1006-07 option B: relief that peels only what pays, then retry once.
        gained = _evict_wholly_below_cap(
            tree_cache, allocator, max(1, int(extend_num_tokens) - payable_size(allocator))
        )
        if gained > 0:
            _flush_deferred_frees(allocator)
            if backup_state:
                state = allocator.backup_state()
            out_cache_loc = _attempt_alloc()
'''

# --- C is the product (``_cap_overask``); the base is the product with C reverted -------------
PRODUCT_TOP = (
    "    evict_from_tree_cache(\n"
    "        tree_cache, num_tokens + _cap_overask(tree_cache, allocator, num_tokens)\n"
    "    )\n"
    "    delivered = max(0, payable_size(allocator) - payable_before)\n"
    "    evict_asked = max(0, num_tokens - payable_before)\n"
)


def _base_source() -> str:
    """alloc_paged_token_slots_extend as it was before C: the product source with the C call
    reverted (loud if the product's call site changes)."""
    from sglang.srt.mem_cache import common as mc

    src = inspect.getsource(mc.alloc_paged_token_slots_extend)
    assert src.count(PRODUCT_TOP) == 1, "product call site of C not found"
    return src.replace(PRODUCT_TOP, ANCHOR_TOP, 1)


def _patched(helper: str = "", hunk_before_rule3: str = "", top: str = ""):
    """The BASE's alloc_paged_token_slots_extend (product minus C) with a draft applied."""
    from sglang.srt.mem_cache import common as mc

    src = _base_source()
    if hunk_before_rule3:
        assert src.count(ANCHOR_RULE3) == 1, "draft anchor RULE3 not found: the draft no longer fits the base"
        src = src.replace(ANCHOR_RULE3, hunk_before_rule3 + ANCHOR_RULE3, 1)
    if top:
        assert src.count(ANCHOR_TOP) == 1, "draft anchor TOP not found: the draft no longer fits the base"
        src = src.replace(ANCHOR_TOP, top, 1)
    g = dict(vars(mc))
    if helper:
        exec(compile(helper, "<draft-helper>", "exec"), g)
    exec(compile(src, "<draft-alloc_paged_token_slots_extend>", "exec"), g)
    return g["alloc_paged_token_slots_extend"]


def _product():
    from sglang.srt.mem_cache import common as mc

    return mc.alloc_paged_token_slots_extend


def _base():
    return _patched()


@pytest.fixture(autouse=True)
def _empty_relief_registry():
    from sglang.srt.mem_cache import common as mc

    mc.clear_extend_relief_providers()
    yield
    mc.clear_extend_relief_providers()


# ---------------------------------------------------------------------------
# 1. the state, pinned on the PRODUCT (green on the base: these document today)
# ---------------------------------------------------------------------------
def test_01_cand3_state_of_the_log_and_the_death_on_the_product(caplog):
    """The double reproduces the numbers of D.log 23:45:03 and the product dies on them."""
    from sglang.srt.mem_cache import common as mc

    pool, tree = _cand3_rank0()
    assert pool.available_size() == 13 * PAGE == 832
    assert tree.evictable_size() == 19200
    assert tree.above_tokens(CAP_PAGES) == 258 * PAGE            # B pays the pool nothing
    caplog.set_level(logging.INFO)
    ok, msg = _try(_base(), tree)
    assert not ok, "the base must die on the cand3 state (the net is empty)"
    assert "Prefill out of memory" in msg
    assert "EVICTION UNDER-DELIVERED" in msg and "the pool received 64" in msg, msg
    assert "A RESIDENCY CAP IS ENGAGED" in msg
    assert any("NO relief provider is registered" in r.getMessage() for r in caplog.records)
    # nf-next-1006-13: the line separates "no provider" from "no net" (Option C is a net, not a provider)
    # and no longer claims the admission count ignores the cap.
    relief_lines = [r.getMessage() for r in caplog.records if "NO relief provider is registered" in r.getMessage()]
    assert all("not a provider" in m and "does not subtract" not in m for m in relief_lines), relief_lines
    assert tree.evict_calls == 1, "the paged path makes ONE peel and never a second"
    assert mc._attempt_extend_relief(1) == 0


def test_02_f7_the_confiscation_round_declines_under_the_published_floor():
    """F7 as the order states it is INERT on NF-D: NF-D publishes the floor every iteration
    (#1045: 351 ``FLOOR PUBLISHED`` lines in D.log, 0 ``peeling further delivered`` lines), and
    _evict_past_confiscation returns 0 as soon as ``uniform_avail_floor`` is set."""
    from sglang.srt.mem_cache import common as mc

    pool, tree = _cand3_rank0()
    _try(_base(), tree)                         # the first peel as the base makes it
    pool_f, tree_f = pool, tree
    tree_f.uniform_avail_floor = 832
    assert mc._evict_past_confiscation(tree_f, pool_f, 1636 - 64) == 0
    assert tree_f.evict_calls == 1, "declined before touching the tree"
    tree_f.uniform_avail_floor = None            # without the floor the round does pay
    assert mc._evict_past_confiscation(tree_f, pool_f, 1636 - 64) >= 1572


@pytest.mark.xfail(strict=True, reason="F7: the paged-extend path never calls the #790 round "
                   "(common.py alloc_token_slots only); flips when option A is built")
def test_03_paged_extend_path_calls_a_confiscation_round():
    from sglang.srt.mem_cache import common as mc

    assert "_evict_past_confiscation" in inspect.getsource(mc.alloc_paged_token_slots_extend)


def test_04_the_relief_seam_is_empty_by_default():
    from sglang.srt.mem_cache import common as mc

    assert mc._extend_relief_providers == []


# ---------------------------------------------------------------------------
# 2. option A0 (F7 literal): the call is added, and nothing changes on NF-D
# ---------------------------------------------------------------------------
def test_05_A0_literal_f7_still_dies_under_the_floor_and_pays_without_it():
    fn = _patched(hunk_before_rule3=DRAFT_A0_HUNK)
    pool, tree = _cand3_rank0()
    ok, msg = _try(fn, tree)
    assert not ok and "the pool received 64" in msg, "A0 with the NF-D floor published: dead code"
    pool2, tree2 = _cand3_rank0()
    tree2.uniform_avail_floor = None            # what the order assumed (floor None)
    ok2, msg2 = _try(fn, tree2)
    assert ok2, msg2


# ---------------------------------------------------------------------------
# 3. option A1: peel through (floor-compatible)
# ---------------------------------------------------------------------------
def test_06_A1_cand3_is_paid_after_peeling_through_B():
    fn = _patched(helper=DRAFT_A1_HELPER, hunk_before_rule3=DRAFT_A1_HUNK)
    pool, tree = _cand3_rank0()
    ok, msg = _try(fn, tree)
    assert ok, msg
    assert tree.evicted == ["L0", "B", "A"], tree.evicted     # B pays nothing, A (below) pays 41 pages


def test_07_A1_chain_case_is_paid_where_B_below_first_cannot():
    a1 = _patched(helper=DRAFT_A1_HELPER, hunk_before_rule3=DRAFT_A1_HUNK)
    pool, tree = _chain_rank0()
    ok, msg = _try(a1, tree)
    assert ok, msg
    assert tree.evicted[:3] == ["X", "B", "A"]


def test_08_A1_no_engaged_cap_no_extra_peel_byte_identical():
    """27B / P / D without a cap: the draft never runs its loop; the failure is the base's."""
    a1 = _patched(helper=DRAFT_A1_HELPER, hunk_before_rule3=DRAFT_A1_HUNK)
    # a pool that is simply full and no cap: the product raises, the draft raises the same
    layout = {"N": (None, _ids(1, 5), 1)}
    pb, tb = _world(layout, free_below=_ids(6, 8), cap=False)
    pa, ta = _world(layout, free_below=_ids(6, 8), cap=False)
    ok_b, msg_b = _try(_base(), tb, 4096)
    ok_a, msg_a = _try(a1, ta, 4096)
    assert not ok_b and not ok_a and msg_a == msg_b
    assert ta.evict_calls == tb.evict_calls == 1


def test_09_A1_a_healthy_allocation_pays_nothing_for_the_draft():
    a1 = _patched(helper=DRAFT_A1_HELPER, hunk_before_rule3=DRAFT_A1_HUNK)
    pool, tree = _cand3_rank0()
    ok, _ = _try(a1, tree, 64 * 5)               # fits the 13 free pages
    assert ok and tree.evict_calls == 0, "no peel at all when the floor-trigger does not fire"
    # (floor 832 >= 64*5+64)


def test_10_A1_rank_local_entry_makes_the_radix_replicas_diverge():
    """THE HAZARD. Ranks 0 and 1 fail the allocation (ids above their cap), rank 2 does not
    (its ids are below). A1 peels A only on the failing ranks: the trees stop being replicas
    -> rank-dependent match_prefix -> rank-dependent extend lengths (#616 / #996 wedge)."""
    a1 = _patched(helper=DRAFT_A1_HELPER, hunk_before_rule3=DRAFT_A1_HUNK)
    worlds = [_cand3_rank0(), _cand3_rank1(), _cand3_rank2()]
    outcome = [_try(a1, t)[0] for (_, t) in worlds]
    assert outcome == [True, True, True]          # every rank allocates ...
    sets = [tuple(t.evicted) for (_, t) in worlds]
    assert len(set(sets)) > 1, sets               # ... and the trees are no longer the same
    assert sets[0] == sets[1] == ("L0", "B", "A") and sets[2] == ("L0", "B"), sets


def test_11_A1_the_divergence_is_real_when_the_first_peel_stops_short_of_A():
    """Same hazard in the chain world: rank 0 fails (X absorbs the peel, B above), rank 2 has
    its ids below and pays from X: A survives on rank 2 and is gone on rank 0."""
    a1 = _patched(helper=DRAFT_A1_HELPER, hunk_before_rule3=DRAFT_A1_HUNK)
    w0 = _chain_rank0()
    w2 = _world({"X": (None, _ids(100, 139), 1), "A": (None, _ids(422, 462), 2),
                 "B": ("A", _ids(140, 357), 3)}, free_below=_ids(408, 420))
    assert _try(a1, w0[1])[0] and _try(a1, w2[1])[0]
    assert "A" in w0[1].evicted and "A" not in w2[1].evicted


# ---------------------------------------------------------------------------
# 4. option B: below-first provider
# ---------------------------------------------------------------------------
def test_12_B_pays_cand3_once_the_first_peel_has_taken_the_leaf_B():
    fn = _patched(helper=DRAFT_B_HELPER, hunk_before_rule3=DRAFT_B_HUNK)
    pool, tree = _cand3_rank0()
    ok, msg = _try(fn, tree)
    assert ok, msg
    assert tree.evicted == ["L0", "B", "A"]


def test_13_B_cannot_reach_a_payable_node_behind_an_above_cap_leaf():
    """The chain world: nothing wholly below the cap is a leaf. B pays 0 and the batch dies."""
    fn = _patched(helper=DRAFT_B_HELPER, hunk_before_rule3=DRAFT_B_HUNK)
    pool, tree = _chain_rank0()
    ok, msg = _try(fn, tree)
    assert not ok and "Prefill out of memory" in msg
    assert tree.evicted == ["X"], tree.evicted     # the draft evicted nothing more


def test_14_B_keeps_the_above_cap_cache_that_A1_throws_away():
    """B's one advantage (a world where the below-cap leaves exist): it peels only what pays."""
    layout = {"X": (None, _ids(900, 939), 1), "Y": (None, _ids(422, 470), 2),
              "Z": (None, _ids(950, 999), 3)}
    for name, helper, hunk in (("B", DRAFT_B_HELPER, DRAFT_B_HUNK), ("A1", DRAFT_A1_HELPER, DRAFT_A1_HUNK)):
        fn = _patched(helper=helper, hunk_before_rule3=hunk)
        pool, tree = _world(layout, free_below=_ids(408, 420))
        ok, msg = _try(fn, tree)
        assert ok, (name, msg)
        assert tree.evicted[:2] == ["X", "Y"], (name, tree.evicted)
        survives_z = "Z" not in tree.evicted
        assert survives_z, (name, tree.evicted)   # both stop once the shortfall is paid here


# ---------------------------------------------------------------------------
# 5. option C = the product: uniform over-ask before the allocation
# ---------------------------------------------------------------------------
def _publish_gap(worlds, cap_pages=CAP_PAGES) -> int:
    """What 01's reduce publishes, through the REAL ``publish_cap_blind_gap``: the group's MIN
    admission is ``MIN_s(A_s + E - U_s)``, every rank gets the same gap. With equal A_s it equals
    max_s U_s (here 259 pages)."""
    from sglang.srt.mem_cache import common as mc

    floor = min(t.uniform_avail_floor for (_, t) in worlds)
    e = max(t.evictable_size() for (_, t) in worlds)
    min_adm = min(p.available_size() + t.evictable_size() - t.above_tokens(cap_pages)
                  for (p, t) in worlds)
    for (p, t) in worlds:
        mc.publish_cap_blind_gap(
            t, allocator=p, pin_admission=True, min_avail=floor, min_admission=min_adm,
            local_evict_full=t.evictable_size(), local_cap_above=t.above_tokens(cap_pages))
    gaps = {mc.cap_blind_gap_published(t) for (_, t) in worlds}
    assert len(gaps) == 1, gaps
    assert e >= 0
    return gaps.pop()


def test_15_C_every_rank_pays_and_the_replicas_stay_identical():
    worlds = [_cand3_rank0(), _cand3_rank1(), _cand3_rank2()]
    assert _publish_gap(worlds) == 259 * PAGE == 16576
    fn = _product()
    results = [_try(fn, t) for (_, t) in worlds]
    assert [r[0] for r in results] == [True, True, True], results
    sets = [tuple(t.evicted) for (_, t) in worlds]
    assert len(set(sets)) == 1, sets                     # REPLICA INVARIANT: same victims everywhere
    assert [t.evict_calls for (_, t) in worlds] == [1, 1, 1], "no collective, no second peel"


def test_16_C_chain_world_is_paid_where_B_dies():
    pool, tree = _chain_rank0()
    _publish_gap([(pool, tree)])
    ok, msg = _try(_product(), tree)
    assert ok, msg


def test_17_C_without_the_published_value_it_is_byte_identical_to_the_base():
    """No gap (01 not publishing, or a rank that cannot publish): the same death, the same message."""
    pb, tb = _cand3_rank0()
    pc, tc = _cand3_rank0()
    ok_b, msg_b = _try(_base(), tb)
    ok_c, msg_c = _try(_product(), tc)
    assert not ok_b and not ok_c and msg_b == msg_c
    assert tb.evicted == tc.evicted and tb.evict_calls == tc.evict_calls == 1


@pytest.mark.parametrize("junk", [0, -5, True, 1.5, "512", None])
def test_17b_C_ignores_a_gap_that_is_not_a_positive_int(junk):
    """01's reader contract (_published_nonneg_int): bool / float / str / negative read as 0."""
    pb, tb = _cand3_rank0()
    pc, tc = _cand3_rank0()
    tc.uniform_cap_blind_gap = junk
    ok_b, msg_b = _try(_base(), tb)
    ok_c, msg_c = _try(_product(), tc)
    assert not ok_b and not ok_c and msg_b == msg_c and tb.evicted == tc.evicted


def test_18_C_no_engaged_cap_asks_for_exactly_what_the_base_asks():
    """27B / P: no KvRowCap engaged -> no over-ask even with a published value."""
    layout = {"N": (None, _ids(1, 5), 1)}
    pb, tb = _world(layout, free_below=_ids(6, 8), cap=False)
    pc, tc = _world(layout, free_below=_ids(6, 8), cap=False)
    tc.uniform_cap_blind_gap = 10 ** 6
    ok_b, msg_b = _try(_base(), tb, 4096)
    ok_c, msg_c = _try(_product(), tc, 4096)
    assert not ok_b and not ok_c and msg_b == msg_c and tb.evicted == tc.evicted


def test_18b_C_a_released_cap_is_no_cap():
    """The cap was engaged and then released (``engaged`` False): the over-ask is off again."""
    from sglang.srt.mem_cache import common as mc

    pool, tree = _cand3_rank0()
    tree.uniform_cap_blind_gap = 16576
    assert mc._cap_overask(tree, pool, 1636) == 16576
    pool._weg2_kv_stage_cap.release()
    assert mc._cap_overask(tree, pool, 1636) == 0


def test_19_C_no_over_ask_when_the_floor_trigger_does_not_fire():
    pool, tree = _cand3_rank0()
    tree.uniform_cap_blind_gap = 16576
    tree.uniform_avail_floor = 10 ** 6                    # plenty: the base would not evict either
    ok, _ = _try(_product(), tree, 64 * 5)
    assert ok and tree.evict_calls == 0


def test_19b_C_the_over_ask_is_exactly_the_published_gap_at_the_trigger():
    """Unit value of ``_cap_overask``: gap when the base's trigger fires, else 0; the ask the
    tree receives is num_tokens + gap (checked at the evict call)."""
    from sglang.srt.mem_cache import common as mc

    pool, tree = _cand3_rank0()
    tree.uniform_cap_blind_gap = 1024
    assert mc._cap_overask(tree, pool, 1636) == 1024          # floor 832 < 1636
    assert mc._cap_overask(tree, pool, 832) == 0              # floor 832 >= 832: base would not evict
    asks = []
    orig = tree.evict
    tree.evict = lambda params: (asks.append(int(params.num_tokens)), orig(params))[1]
    _try(_product(), tree)
    assert asks[0] == 1636 + 1024, asks                       # tree.evict(num_tokens + gap)
    asks.clear()
    pb, tb = _cand3_rank0()
    orig_b = tb.evict
    tb.evict = lambda params: (asks.append(int(params.num_tokens)), orig_b(params))[1]
    _try(_base(), tb)
    assert asks == [1636], asks                               # the base asks exactly num_tokens


def test_20_C_cost_is_the_over_eviction_in_cache_tokens():
    """The price, as a number: tokens the tree gives up beyond the base's one peel (cand3 world)."""
    pb, tb = _cand3_rank0()
    _try(_base(), tb)
    base_evicted = 19200 - tb.evictable_size()
    pc, tc = _cand3_rank0()
    _publish_gap([(pc, tc)])
    assert _try(_product(), tc)[0]
    over = (19200 - tc.evictable_size()) - base_evicted
    assert (base_evicted, over) == (16576, 2624), (base_evicted, over)


# ---------------------------------------------------------------------------
# 6. the red/green matrix: "the cand3 / chain state allocates without a RuntimeError"
#    (column "base" = the base without C = the expected red of the 07 report; "product" = C)
# ---------------------------------------------------------------------------
def _variant(name: str):
    return {
        "base": _base,
        "A0": lambda: _patched(hunk_before_rule3=DRAFT_A0_HUNK),
        "A1": lambda: _patched(helper=DRAFT_A1_HELPER, hunk_before_rule3=DRAFT_A1_HUNK),
        "B": lambda: _patched(helper=DRAFT_B_HELPER, hunk_before_rule3=DRAFT_B_HUNK),
        "product": _product,
    }[name]()


_RED = {
    ("cand3", "base"): "the net is empty (cand3 05.10. 23:45:03Z)",
    ("cand3", "A0"): "F7 literal declines under the published floor",
    ("chain", "base"): "the net is empty",
    ("chain", "A0"): "F7 literal declines under the published floor",
    ("chain", "B"): "no leaf wholly below the cap: below-first pays 0",
}


@pytest.mark.parametrize("variant", ["base", "A0", "A1", "B", "product"])
@pytest.mark.parametrize("state", ["cand3", "chain"])
def test_21_matrix_allocation_survives_the_gap(state, variant):
    pool, tree = _cand3_rank0() if state == "cand3" else _chain_rank0()
    _publish_gap([(pool, tree)])
    ok, msg = _try(_variant(variant), tree)
    expect_ok = (state, variant) not in _RED
    assert ok == expect_ok, (state, variant, _RED.get((state, variant)), msg[:200])


def test_22_a_cap_aware_admission_admits_cand3_and_the_peel_still_does_not_reach():
    """WHY 01 ALONE DOES NOT CLOSE THE CLASS. nf-next-1006-01 counts evictable minus the tokens
    above the cap (budget = avail + E - U). In the cand3 state that is 832 + 19200 - 16512 = 3520
    >= 1572: the request IS fundable in principle (A's 41 pages + L0 + the 13 free pages sit below
    the cap), so the cap-aware admission lets it through -- and the one peel the paged path makes
    is absorbed by the atomic above-cap leaf B (16512 tokens >= the ask 1636) before it reaches A.
    Count (admission) and reach (peel order) are different questions."""
    pool, tree = _cand3_rank0()
    e = tree.evictable_size()
    u = tree.above_tokens(CAP_PAGES)
    cap_aware_budget = pool.available_size() + e - u
    assert (e, u, cap_aware_budget) == (19200, 258 * PAGE, 3520)
    assert cap_aware_budget >= EXTEND, "the cap-aware admission admits"
    ok, msg = _try(_base(), tree)
    assert not ok and "the pool received 64" in msg, "and the allocation site (base) still dies"
    assert tree.evicted == ["L0", "B"], "A, the payable node, was never reached"
