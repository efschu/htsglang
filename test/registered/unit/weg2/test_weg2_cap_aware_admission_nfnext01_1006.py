"""nf-next-1006-01: the admission counts only evictable tokens BELOW the engaged residency cap.

THE DEATH (cand3 bce16a6ddf, D.log 05.10. 23:45:03Z, TP0). weg2-62-415: the host load-back took
26048 of the 26880 free tokens under the cap, ``full_available_size=832``, and the admission still
saw "Available full tokens: 20032" = 832 + 19200 evictable. The 19200 sit ABOVE the cap
(``KvRowCap``): the pool takes every id the peel frees up there straight back, so the pool
delivered 64 of the 868 asked ("EVICTION UNDER-DELIVERED ... A RESIDENCY CAP IS ENGAGED") and
all three D ranks died. The lift tick (``_group_room_below``) had never counted evictable at all;
tick and admission contradicted each other. 1540 gave the tick the host load-back; this closes the
admission side.

WHAT THE FIX DOES (pinned here, CPU only, real code, doubles for pool/tree):
  * ``Scheduler._update_uniform_pool_budget`` (the reduce that already runs once per iteration)
    subtracts this rank's evictable-above-cap from the ``local_admission`` vote it already
    contributes under uneven DCP -> the group MIN is cap-aware, ``dcp_avail_deficit`` carries it
    to every rank, ``PrefillAdder.rem_total_tokens`` / ``cur_rem_tokens`` subtract the SAME
    number (it cancels in the deficit), and ``fundable_extend_tokens`` is cut by the group gap
    (floor + E - MIN admission, built from reduced values only).
  * No new collective site, no new payload element, no new guard in front of a collective, no env.
  * Only behind an engaged ``KvRowCap`` AND the existing uneven-DCP pin; otherwise nothing is
    published and every reader returns what it returned before.
  * Instrument ``WEG2 CAP-BLIND-ADMIT rid= avail= evictable_reported= above_cap= grant=``
    (<= 1 per second per rank, only when above_cap > 0 and the admission was cut).
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import ast  # noqa: E402
import logging  # noqa: E402
import pathlib  # noqa: E402
import types  # noqa: E402
import unittest.mock as um  # noqa: E402
from typing import List  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

import sys  # noqa: E402

# the three-rank reduce harness lives in the managers test dir
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "managers"))

from test_collective_family_siblings_610 import (  # noqa: E402
    BudgetHarness,
    ThreadCollective,
    _process_patches,
    run_ranks,
)

PAGE = 64
NUM_PAGES = 4096
CAP_PAGES = 512                      # S0 = 32768 tokens
NRANKS = 3
CHUNK = 4096


# ---------------------------------------------------------------------------
# doubles
# ---------------------------------------------------------------------------
class _Pool:
    """Paged allocator double: 1-based page ids, free lists as tensors, the free
    listener the real ``KvRowCap`` subscribes to."""

    page_size = PAGE
    num_pages = NUM_PAGES

    def __init__(self, free_ids: List[int]):
        self.free_pages = torch.tensor(sorted(free_ids), dtype=torch.int64)
        self.release_pages = torch.empty(0, dtype=torch.int64)
        self.free_group: list = []
        self._on_free: list = []
        self._on_clear: list = []
        self.residency_withheld_slots = 0

    def register_free_listener(self, on_free, on_clear=None):
        self._on_free.append(on_free)
        if on_clear is not None:
            self._on_clear.append(on_clear)

    def available_size(self) -> int:
        return (int(self.free_pages.numel()) + int(self.release_pages.numel())) * PAGE


class _CD:
    def __init__(self, value):
        self.value = value
        self.lock_ref = 0


class _Node:
    def __init__(self, value=None):
        self.children = {}
        self.evicted = False
        self.component_data = [_CD(value)]


class _Tree:
    """Radix-tree double: one node per leaf, ``value`` = the token slots of that leaf. Exposes the
    surface ``fundable_extend_tokens``, the reduce and ``above_cap_tokens`` read. Has NO
    ``deliverable_evictable_size`` (the fallback count answers, like every duck-typed stand-in)."""

    uniform_avail_floor = None
    uniform_admitted_since_floor = 0

    def __init__(self, pool: _Pool, leaf_pages: List[int]):
        self.token_to_kv_pool_allocator = pool
        self.root_node = _Node()
        for i, p in enumerate(leaf_pages):
            slots = torch.arange(p * PAGE, (p + 1) * PAGE, dtype=torch.int64)
            self.root_node.children[i] = _Node(slots)
        self._n = len(leaf_pages)

    def _collect_all_nodes(self):
        return [self.root_node] + list(self.root_node.children.values())

    def evictable_size(self) -> int:
        return self._n * PAGE

    full_evictable_size = evictable_size

    def is_chunk_cache(self):
        return False


def _engage(pool: _Pool, cap_pages: int = CAP_PAGES):
    """The real stage cap on the allocator double (what ``_engage_kv_cap`` leaves)."""
    from sglang.srt.weg2 import d_seat_vram as dsv

    return dsv._engage_kv_cap(pool, cap_pages * PAGE, PAGE)


def _rank_state(*, free_below: int, above: int, below: int, engage: bool = True):
    """A rank: ``free_below`` free pages under the cap (ids 1..), ``above`` evictable leaves above
    the cap (ids cap+1..), ``below`` evictable leaves below it (ids after the free ones)."""
    free_ids = list(range(1, free_below + 1))
    leaf_ids = list(range(free_below + 1, free_below + 1 + below)) + list(
        range(CAP_PAGES + 1, CAP_PAGES + 1 + above)
    )
    # ids above the cap that are NOT leaves are free in the pool; the real cap withholds them
    above_free = [i for i in range(CAP_PAGES + 1, NUM_PAGES + 1) if i not in set(leaf_ids)]
    pool = _Pool(free_ids + above_free)
    tree = _Tree(pool, leaf_ids)
    if engage:
        _engage(pool)
    return pool, tree


# cand3 numbers scaled to page 64: TP0 832 free below the cap, 19200 evictable all above it
RANKS = [
    dict(free_below=13, above=300, below=0),      # 832 free, 19200 above  (the TP0 of the death)
    dict(free_below=20, above=187, below=113),    # 1280 free, 11968 above, 7232 below
    dict(free_below=14, above=300, below=0),      # 896 free, 19200 above
]


class _CapHarness(BudgetHarness):
    """The real reduce, per rank, with this rank's pool and tree."""

    def __init__(self, collective, pool, tree, dcp_size=NRANKS, tp_rank=0):
        super().__init__(collective, 0, 0, dcp_size=dcp_size, tp_rank=tp_rank)
        self.token_to_kv_pool_allocator = pool
        self.tree_cache = tree
        self.page_size = PAGE


def _run_reduce(states, *, pin: bool = True):
    """Run the production reduce on ``len(states)`` rank threads; return the harnesses."""
    collective = ThreadCollective(len(states))
    harnesses: list = [None] * len(states)

    def body(rank):
        pool, tree = states[rank]
        h = _CapHarness(collective, pool, tree, dcp_size=len(states), tp_rank=rank)
        h._update_uniform_pool_budget()
        harnesses[rank] = h
        return h

    with _process_patches(collective, um.patch(
            "sglang.srt.distributed.utils.uneven_dcp_active", lambda *a: pin)):
        _res, errors = run_ranks(body, nranks=len(states))
    collective.abort()
    assert errors == [], errors
    return harnesses


def _adder(h, tree, *, offset: int = 0):
    """A ``PrefillAdder`` past ``__init__`` with the state the two budget properties read."""
    from sglang.srt.managers.schedule_policy import PrefillAdder
    from sglang.srt.mem_cache.common import published_fundable_floor
    from sglang.srt.planner.chunked_admission import ChunkedCommitmentLedger

    a = PrefillAdder.__new__(PrefillAdder)
    a.is_all_swa = False
    a.is_hybrid_swa = False
    a.is_hybrid_ssm_cache = True            # the NF branch: deliverable evictable
    a.token_to_kv_pool_allocator = tree.token_to_kv_pool_allocator
    a.tree_cache = tree
    a.rem_total_token_offset = offset
    a.cur_rem_token_offset = offset
    a.dcp_avail_deficit = h.uniform_budget_deficit()
    a.fundable_extend_floor = published_fundable_floor(tree)
    a.chunked_admission_enabled = True
    a.commitment_ledger = ChunkedCommitmentLedger()
    a.page_size = PAGE
    a.rem_chunk_tokens = CHUNK
    return a


def _budgets(states, *, pin=True):
    from sglang.srt.managers.schedule_policy import PrefillAdder

    hs = _run_reduce(states, pin=pin)
    out = []
    for h, (pool, tree) in zip(hs, states):
        a = _adder(h, tree)
        out.append(dict(
            rem_total=int(PrefillAdder.rem_total_tokens.fget(a)),
            cur_rem=int(PrefillAdder.cur_rem_tokens.fget(a)),
            fundable=_fundable(tree),
            deficit=h.uniform_budget_deficit(),
            floor=tree.uniform_avail_floor,
        ))
    return hs, out


def _fundable(tree) -> int:
    from sglang.srt.mem_cache.common import fundable_extend_tokens

    return fundable_extend_tokens(tree)


def _grant(fundable: int, want: int = 1572) -> int:
    from sglang.srt.mem_cache.common import chunk_tokens_the_pool_can_fund

    return min(want, chunk_tokens_the_pool_can_fund(fundable, PAGE, CHUNK))


# ---------------------------------------------------------------------------
# (1) the death, rebuilt: avail 832, evictable 19200 above the cap, request 1572 -> budget 832
# ---------------------------------------------------------------------------
def test_death_numbers_budget_and_grant_are_832():
    states = [_rank_state(**r) for r in RANKS]
    assert states[0][0].available_size() == 832
    hs, b = _budgets(states)
    assert b[0]["floor"] == 832, b                       # group MIN of the free tokens under the cap
    for i in range(NRANKS):
        assert b[i]["fundable"] == 832, (i, b[i])        # not 20032
        assert b[i]["rem_total"] == 832, (i, b[i])
        assert b[i]["cur_rem"] == 832, (i, b[i])
    assert _grant(b[0]["fundable"]) == 832               # 13 pages; the rest runs as a follow chunk


def test_death_numbers_blind_formula_would_say_20032():
    """The rebuilt state is the death's: the cap-blind count of TP0 is the 20032 of the log."""
    states = [_rank_state(**r) for r in RANKS]
    pool, tree = states[0]
    assert pool.available_size() + tree.evictable_size() == 20032


# ---------------------------------------------------------------------------
# (3) three ranks: the verdict is rank-gleich, no new collective
# ---------------------------------------------------------------------------
def test_three_ranks_agree_on_every_admission_number():
    states = [_rank_state(**r) for r in RANKS]
    hs, b = _budgets(states)
    for key in ("rem_total", "cur_rem", "fundable"):
        assert len({x[key] for x in b}) == 1, (key, [x[key] for x in b])
    # and a per-rank number that DIFFERS (the rank-local part) is what the deficit carries
    assert len({x["deficit"] for x in b}) > 1


def test_three_ranks_agree_with_a_running_reservation_charged():
    from sglang.srt.managers.schedule_policy import PrefillAdder

    states = [_rank_state(**r) for r in RANKS]
    hs = _run_reduce(states)
    vals = []
    for h, (pool, tree) in zip(hs, states):
        a = _adder(h, tree, offset=200)
        vals.append((int(PrefillAdder.rem_total_tokens.fget(a)), int(PrefillAdder.cur_rem_tokens.fget(a))))
    assert len(set(vals)) == 1, vals
    assert vals[0] == (632, 632)


def test_the_reduce_takes_one_collective_of_the_same_width_on_every_rank():
    states = [_rank_state(**r) for r in RANKS]
    collective = ThreadCollective(NRANKS)
    calls = [0] * NRANKS
    widths: list = [[] for _ in range(NRANKS)]
    real = collective.all_reduce

    def counting(tensor, op=None, group=None):
        import threading

        r = threading.current_thread().rank
        calls[r] += 1
        widths[r].append(int(tensor.numel()))
        return real(tensor, op=op, group=group)

    collective.all_reduce = counting

    def body(rank):
        pool, tree = states[rank]
        _CapHarness(collective, pool, tree, dcp_size=NRANKS, tp_rank=rank)._update_uniform_pool_budget()

    with _process_patches(collective, um.patch(
            "sglang.srt.distributed.utils.uneven_dcp_active", lambda *a: True)):
        _res, errors = run_ranks(body, nranks=NRANKS)
    collective.abort()
    assert errors == [], errors
    assert calls == [calls[0]] * NRANKS and calls[0] >= 1
    assert widths[0] == widths[1] == widths[2]


def test_no_collective_site_was_added_to_the_scheduler():
    """P1 of the 1537 pin, restated for this change: the group collectives of scheduler.py by
    (function, call) are exactly the 1537 table (bce16a6ddf) -- this change added none."""
    import sglang.srt.managers.scheduler as sched_mod

    src = pathlib.Path(sched_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    ops = {"all_reduce", "all_gather", "all_gather_object", "broadcast", "broadcast_object_list",
           "barrier", "reduce_scatter", "gather", "scatter", "reduce", "send", "recv",
           "broadcast_pyobj", "all_gather_into_tensor", "reduce_scatter_tensor", "batch_isend_irecv"}
    found: dict = {}

    def walk(node, stack):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                walk(child, stack + [child.name])
                continue
            if isinstance(child, ast.ClassDef):
                walk(child, stack + [child.name])
                continue
            if isinstance(child, ast.Call):
                f = child.func
                name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                if name in ops:
                    chain = ast.unparse(f)
                    if chain.startswith("torch.distributed") or name in ("broadcast_pyobj",):
                        key = (".".join(stack), chain if chain.startswith("torch") else name)
                        found[key] = found.get(key, 0) + 1
            walk(child, stack)

    walk(tree, [])
    expected = {
        ("Scheduler._uniform_timeout_ballot", "torch.distributed.all_reduce"): 1,
        ("Scheduler._form_a_tp_exchange", "broadcast_pyobj"): 1,
        ("Scheduler._form_a_tp_gather", "torch.distributed.all_gather_object"): 1,
        ("Scheduler._process_and_broadcast_mm_inputs", "torch.distributed.broadcast_object_list"): 2,
        ("Scheduler._weg2_group_min_flags", "torch.distributed.all_reduce"): 1,
        ("Scheduler._weg2_group_min_ints", "torch.distributed.all_reduce"): 1,
        ("Scheduler._update_uniform_pool_budget", "torch.distributed.all_reduce"): 2,
        ("Scheduler.handle_rpc_request", "torch.distributed.barrier"): 1,
    }
    # the barrier is spelled `barrier` through another alias in the 1537 table; compare the rest
    got = {k: v for k, v in found.items() if "barrier" not in k[1]}
    want = {k: v for k, v in expected.items() if "barrier" not in k[1]}
    assert got == want, (sorted(got.items()), sorted(want.items()))


# ---------------------------------------------------------------------------
# (2) "Flip unveraendert": nothing published, every number as before
# ---------------------------------------------------------------------------
def test_without_an_engaged_cap_nothing_changes_27b_flip_p():
    """P, the 27B flip and a D without a stage form never create the cap: the allocator carries no
    ``_weg2_kv_stage_cap``. The same pools, the same leaves -> the pre-F1 numbers."""
    states = [_rank_state(**r, engage=False) for r in RANKS]
    for pool, _tree in states:
        assert getattr(pool, "_weg2_kv_stage_cap", None) is None
    hs, b = _budgets(states)
    floor = min(p.available_size() for p, _t in states)
    pinned = min(p.available_size() + t.evictable_size() for p, t in states)
    for i, x in enumerate(b):
        assert x["fundable"] == floor + 19200, (i, x)        # floor MIN + replicated E
        assert x["rem_total"] == pinned, (i, x)
        assert getattr(states[i][1], "uniform_cap_blind_gap", 0) == 0
        assert getattr(states[i][1], "uniform_cap_above_tokens", 0) == 0


def test_a_released_cap_counts_as_none():
    states = [_rank_state(**r) for r in RANKS]
    for pool, _t in states:
        pool._weg2_kv_stage_cap.release()
        assert not pool._weg2_kv_stage_cap.engaged
    hs, b = _budgets(states)
    floor = min(p.available_size() for p, _t in states)
    pinned = min(p.available_size() + t.evictable_size() for p, t in states)
    for x in b:
        assert x["fundable"] == floor + 19200 and x["rem_total"] == pinned, x


def test_without_the_uneven_dcp_pin_nothing_is_published():
    """Even TP / 27B: no ``local_admission`` vote exists, so the rank-local above-cap number could
    not be pinned; it is therefore not used (the admission stays as it was)."""
    states = [_rank_state(**r) for r in RANKS]
    hs, b = _budgets(states, pin=False)
    for i, (x, (pool, tree)) in enumerate(zip(b, states)):
        assert getattr(tree, "uniform_cap_above_tokens", 0) == 0
        assert getattr(tree, "uniform_cap_blind_gap", 0) == 0
        assert x["fundable"] == 832 + 19200, (i, x)            # floor MIN + E, as before


def test_an_engaged_cap_with_nothing_above_it_is_byte_equal():
    """The steady state of a D: the cap is the mapped stage, no evictable leaf lies above it."""
    states = [_rank_state(free_below=f, above=0, below=300) for f in (13, 20, 14)]
    hs, b = _budgets(states)
    for i, x in enumerate(b):
        assert getattr(states[i][1], "uniform_cap_above_tokens", 0) == 0
        assert getattr(states[i][1], "uniform_cap_blind_gap", 0) == 0
        assert x["fundable"] == 13 * PAGE + 300 * PAGE, (i, x)
        assert x["rem_total"] == min(f * PAGE + 300 * PAGE for f in (13, 20, 14)), (i, x)


def test_single_rank_clears_what_a_tp_layout_published():
    from sglang.srt.mem_cache.common import CAP_ABOVE_ATTR, CAP_GAP_ATTR

    pool, tree = _rank_state(**RANKS[0])
    setattr(tree, CAP_ABOVE_ATTR, 19200)
    setattr(tree, CAP_GAP_ATTR, 19200)
    h = _CapHarness(ThreadCollective(1), pool, tree, dcp_size=1)
    h.tp_cpu_group = None       # one rank: the early path
    h.ps = types.SimpleNamespace(tp_rank=0, tp_size=1)
    h._update_uniform_pool_budget()
    assert getattr(tree, CAP_ABOVE_ATTR) == 0 and getattr(tree, CAP_GAP_ATTR) == 0
    assert _fundable(tree) == 832 + 19200


# ---------------------------------------------------------------------------
# unit: the above-cap count (page ids, 1-based; slot // page_size > cap)
# ---------------------------------------------------------------------------
def test_above_cap_counts_pages_strictly_above_the_cap_id():
    from sglang.srt.mem_cache import evict_frontier_census as ef

    pool = _Pool([])
    leaves = [CAP_PAGES - 1, CAP_PAGES, CAP_PAGES + 1, CAP_PAGES + 2]   # the cap page itself is BELOW
    tree = _Tree(pool, leaves)
    assert ef.above_cap_tokens(tree, 0, CAP_PAGES, PAGE) == 2 * PAGE


def test_above_cap_skips_locked_and_evicted_nodes():
    from sglang.srt.mem_cache import evict_frontier_census as ef

    pool = _Pool([])
    tree = _Tree(pool, [CAP_PAGES + 1, CAP_PAGES + 2, CAP_PAGES + 3])
    kids = list(tree.root_node.children.values())
    kids[0].component_data[0].lock_ref = 1
    kids[1].evicted = True
    assert ef.above_cap_tokens(tree, 0, CAP_PAGES, PAGE) == PAGE


def test_the_memo_rescans_on_a_cap_change_and_reuses_within_the_ttl():
    from sglang.srt.mem_cache import evict_frontier_census as ef

    pool, tree = _rank_state(free_below=13, above=10, below=0)
    allocator = pool
    first = ef.cap_above_tokens_memo(tree, 0, allocator, PAGE)
    assert first == 10 * PAGE
    # a leaf peeled: within the TTL the (higher) old value is reused
    del tree.root_node.children[0]
    tree._n -= 1
    assert ef.cap_above_tokens_memo(tree, 0, allocator, PAGE) == first
    # a cap change always rescans
    pool._weg2_kv_stage_cap.release()
    _engage(pool, CAP_PAGES + 4)
    assert ef.cap_above_tokens_memo(tree, 0, allocator, PAGE) <= 9 * PAGE


def test_a_failing_scan_reads_zero_not_an_exception():
    from sglang.srt.mem_cache.common import cap_above_evictable

    pool, tree = _rank_state(**RANKS[0])
    tree._collect_all_nodes = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    assert cap_above_evictable(tree, allocator=pool, page_size=PAGE) == 0


# ---------------------------------------------------------------------------
# (b) the instrument
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _reset_rate_limit():
    from sglang.srt.mem_cache import common

    box = getattr(common, "_cap_blind_last_log", None)   # absent on the 1540 stand
    if box is not None:
        box[0] = 0.0
    yield
    if box is not None:
        box[0] = 0.0


def _stub_adder(tree, pool, *, rem_chunk=CHUNK):
    from sglang.srt.managers.schedule_policy import PrefillAdder

    a = PrefillAdder.__new__(PrefillAdder)
    a.tree_cache = tree
    a.token_to_kv_pool_allocator = pool
    a.page_size = PAGE
    a.rem_chunk_tokens = rem_chunk
    a.lifetime_refusal = None
    return a


def _published(tree, above, gap):
    from sglang.srt.mem_cache.common import CAP_ABOVE_ATTR, CAP_GAP_ATTR

    setattr(tree, CAP_ABOVE_ATTR, above)
    setattr(tree, CAP_GAP_ATTR, gap)


def _lines(caplog):
    return [r.getMessage() for r in caplog.records if "CAP-BLIND-ADMIT" in r.getMessage()]


def test_instrument_line_on_a_cut_chunk_grant(caplog):
    from sglang.srt.managers.schedule_policy import PrefillAdder

    pool, tree = _rank_state(**RANKS[0])
    _published(tree, 19200, 19200)
    a = _stub_adder(tree, pool)
    req = types.SimpleNamespace(rid="weg2-62-415")
    caplog.set_level(logging.WARNING)
    fundable = 832                                     # what the cap-aware count says
    PrefillAdder._note_cap_blind_chunk(a, req, fundable, 832)
    out = _lines(caplog)
    assert len(out) == 1, out
    assert out[0].startswith("WEG2 CAP-BLIND-ADMIT rid=weg2-62-415 avail=832 evictable_reported=19200 "
                             "above_cap=19200 grant=832"), out[0]


def test_instrument_silent_when_the_grant_was_not_cut(caplog):
    from sglang.srt.managers.schedule_policy import PrefillAdder

    pool, tree = _rank_state(**RANKS[0])
    _published(tree, 64, 64)
    a = _stub_adder(tree, pool)
    caplog.set_level(logging.WARNING)
    # fundable 20000 -> grant 4096 either way (the chunk is the binding limit): not cut
    PrefillAdder._note_cap_blind_chunk(a, types.SimpleNamespace(rid="r"), 20000, 4096)
    assert _lines(caplog) == []


def test_instrument_silent_without_anything_above_the_cap(caplog):
    from sglang.srt.managers.schedule_policy import PrefillAdder

    pool, tree = _rank_state(**RANKS[0])
    _published(tree, 0, 0)
    a = _stub_adder(tree, pool)
    caplog.set_level(logging.WARNING)
    PrefillAdder._note_cap_blind_chunk(a, types.SimpleNamespace(rid="r"), 832, 832)
    PrefillAdder._note_cap_blind_refusal(a, types.SimpleNamespace(rid="r"), 5000, 4000)
    assert _lines(caplog) == []


def test_instrument_on_a_refused_new_request(caplog):
    from sglang.srt.managers.schedule_policy import PrefillAdder

    pool, tree = _rank_state(**RANKS[0])
    _published(tree, 19200, 19200)
    a = _stub_adder(tree, pool)
    caplog.set_level(logging.WARNING)
    # refused at budget 832; the blind budget (832 + 19200) would have admitted 5000
    PrefillAdder._note_lifetime_refusal(a, types.SimpleNamespace(rid="n1"), 5000, 832)
    out = _lines(caplog)
    assert len(out) == 1 and "rid=n1" in out[0] and "above_cap=19200" in out[0] and "grant=832" in out[0], out
    assert a.lifetime_refusal == ("n1", 5000, 832)         # the 1531 bookkeeping is untouched


def test_instrument_silent_when_blind_would_have_refused_too(caplog):
    from sglang.srt.managers.schedule_policy import PrefillAdder

    pool, tree = _rank_state(**RANKS[0])
    _published(tree, 100, 100)
    a = _stub_adder(tree, pool)
    caplog.set_level(logging.WARNING)
    PrefillAdder._note_lifetime_refusal(a, types.SimpleNamespace(rid="n2"), 5000, 832)
    assert _lines(caplog) == []


def test_instrument_is_limited_to_one_line_per_second(caplog):
    from sglang.srt.mem_cache import common

    caplog.set_level(logging.WARNING)
    with um.patch("time.monotonic", side_effect=[100.0, 100.4, 101.2]):
        assert common.note_cap_blind_admit(rid="a", avail=1, evictable_reported=2, above_cap=3, grant=4)
        assert not common.note_cap_blind_admit(rid="b", avail=1, evictable_reported=2, above_cap=3, grant=4)
        assert common.note_cap_blind_admit(rid="c", avail=1, evictable_reported=2, above_cap=3, grant=4)
    assert len(_lines(caplog)) == 2
    assert not common.note_cap_blind_admit(rid="d", avail=1, evictable_reported=2, above_cap=0, grant=4)


def test_instrument_never_raises_on_the_gate_path():
    from sglang.srt.managers.schedule_policy import PrefillAdder

    a = PrefillAdder.__new__(PrefillAdder)             # no tree at all
    a.tree_cache = None
    PrefillAdder._note_cap_blind_chunk(a, types.SimpleNamespace(rid="r"), 1, 1)
    PrefillAdder._note_cap_blind_refusal(a, types.SimpleNamespace(rid="r"), 1, 1)


# ---------------------------------------------------------------------------
# the readers are inert without a publication (duck-typed stand-ins, 27B / P trees)
# ---------------------------------------------------------------------------
def test_readers_without_a_publication_are_the_old_functions():
    from sglang.srt.mem_cache import common

    class _T:
        token_to_kv_pool_allocator = types.SimpleNamespace(available_size=lambda: 46)

        def evictable_size(self):
            return 149121

    t = _T()
    assert common.cap_above_published(t) == 0 and common.cap_blind_gap_published(t) == 0
    assert common.fundable_extend_tokens(t) == 46 + 149121
    assert common.deliverable_evictable_cap_aware_or(t, t.evictable_size) == 149121
    m = um.MagicMock()                                   # a duck-typed stand-in answers every getattr
    assert common.cap_above_published(m) == 0 and common.cap_blind_gap_published(m) == 0


def test_cap_aware_count_never_goes_below_zero():
    from sglang.srt.mem_cache import common

    class _T:
        evictable = 100

        def evictable_size(self):
            return self.evictable

    t = _T()
    setattr(t, common.CAP_ABOVE_ATTR, 500)
    assert common.deliverable_evictable_cap_aware_or(t, t.evictable_size) == 0

# ---------------------------------------------------------------------------
# the lifetime gate: the cap-aware budget guards the IMMEDIATE allocation only
# ---------------------------------------------------------------------------
def _waiver(tree, total, born, rem_total):
    from sglang.srt.managers.schedule_policy import PrefillAdder

    a = PrefillAdder.__new__(PrefillAdder)
    a.tree_cache = tree
    return PrefillAdder._cap_lifetime_waiver(a, total, born, rem_total)


def test_waiver_admits_when_only_the_decode_reserve_exceeds_the_cap_aware_budget():
    """room under the cap 1000, request 300 + page 64 immediate, max_new 1024: the pool under the
    cap serves it, and the decode growth is the tick's (a cap-aware LIFETIME gate would park the
    request for as long as the tree holds evictable tokens above the cap, idle server included)."""
    pool, tree = _rank_state(**RANKS[0])
    _published(tree, 19200, 19200)
    assert _waiver(tree, total=364 + 1024, born=364, rem_total=1000) is True


def test_waiver_refuses_the_death_request():
    """weg2-62-415: load-back 26048 + extend 1572 + page against room 26880 -> the immediate part
    does not fit, no waiver, NO_TOKEN (the tick lifts the cap for it, 1540)."""
    pool, tree = _rank_state(**RANKS[0])
    _published(tree, 19200, 19200)
    born = 26048 + 1572 + 64
    assert _waiver(tree, total=born + 1024, born=born, rem_total=26880) is False


def test_waiver_is_off_without_a_gap_and_where_the_blind_budget_refuses_too():
    pool, tree = _rank_state(**RANKS[0])
    _published(tree, 0, 0)
    assert _waiver(tree, total=1388, born=364, rem_total=1000) is False       # as before F1
    _published(tree, 19200, 19200)
    assert _waiver(tree, total=1000 + 19200, born=364, rem_total=1000) is False  # blind refuses too
    assert _waiver(None, total=1388, born=364, rem_total=1000) is False       # no tree: as before


def test_both_lifetime_gates_of_add_one_req_ask_the_waiver():
    from sglang.srt.managers import schedule_policy as sp

    src = open(sp.__file__).read()
    i = src.index("    def add_one_req(")
    j = src.index("\n    def ", i + 10)
    body = src[i:j]
    assert body.count("self._cap_lifetime_waiver(") == 2, "pre-lock and post-lock lifetime gate"
