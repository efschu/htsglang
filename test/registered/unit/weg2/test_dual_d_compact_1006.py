# SPDX-License-Identifier: Apache-2.0
"""D-COMPACT (weg2/dual_d_compact.py, switch SGLANG_WEG2_DUAL_D_COMPACT, default on in the dual D).

Metal dual262kbar1fs10061152 (D 11:59:07-12:06:55): two running decode seats held D rows 198716/198717,
the D-KV span shrinks only from the top, ``SHRINK-BLOCKED reason=live_floor`` 39x, P's grant after 473 s.

Real classes on the CPU: MHATokenToKVPool, TokenToKVPoolAllocator (+ the real KvRowCap through
``engage_cap``), ReqToTokenPool, UnifiedTreeNode, DraftKVSlotMapper, DKvStage + CardKvLedger. Three D ranks
run in threads with a barrier MIN collective (the uneven-DCP owner rule, S=8, ratios 4/2/2).

DANGER DIRECTIONS: a request reads other bytes after the move (wrong row, reference not switched, a rank
skipped, a draft row lost), the group parts company (a rank decides differently -> a hang or a CapBreach),
a half-moved table, the flip form touched. Mutants for each turn the matching check red.
"""
from __future__ import annotations

import os
import tempfile
import threading
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator  # noqa: E402
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool  # noqa: E402
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402
from sglang.srt.mem_cache.unified_radix_cache import UnifiedTreeNode  # noqa: E402
from sglang.srt.speculative.dflash_solo_pool import DraftKVSlotMapper  # noqa: E402
from sglang.srt.weg2 import card_kv_ledger as K  # noqa: E402
from sglang.srt.weg2 import dual_d_compact as C  # noqa: E402
from sglang.srt.weg2 import dual_d_kv_stage as D  # noqa: E402
from sglang.srt.weg2 import dual_p_kv_stage as P  # noqa: E402
from sglang.srt.weg2.d_seat_vram import AllocInfo  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

SIZE = 256                      # global id space (D's allocator, replicated)
STEP = 16
MAPPED = 240
DUAL_D = {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "D", D.MAX_TOKENS_ENV: str(SIZE)}
KEYS = tuple(DUAL_D) + (C.SWITCH_ENV, "SGLANG_WEG2_DUAL_D_LIVE_YIELD_WAIT_S")
PREFIX3 = [0, 4, 6, 8]          # uneven DCP: S=8, rank ratios 4/2/2
G = 4096                        # granule: small, so the stage bytes move with the tokens


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)


def _dual(monkeypatch, **extra):
    for k, v in {**DUAL_D, **extra}.items():
        monkeypatch.setenv(k, v)


# ------------------------------------------------------------------------------------------- world

class _Spans:
    available = True

    def info(self, ptr):
        return AllocInfo(size=(SIZE + 1) * 64, mapped=0, planned=0, active=True)

    def set_spans(self, ptr, spans, now):
        return 0


def _val(gid, layer, kv):
    """The bytes token ``gid`` wrote into layer ``layer`` (k=0 / v=1): exact in float16."""
    return float(gid) + 0.25 * layer + (0.5 if kv else 0.0)


class Rank:
    """One D rank: its compact pool share (or the whole pool), the replicated allocator / table."""

    def __init__(self, r, bounds, prefix, ops):
        self.r, self.bounds, self.prefix = r, bounds, prefix
        if bounds is None:
            rows = SIZE + 1
        else:
            S, lo, hi = bounds
            rows = (SIZE // S + 1) * (hi - lo)
        self.pool = MHATokenToKVPool(size=rows - 1, page_size=1, dtype=torch.float16, head_num=2, head_dim=4,
                                     layer_num=2, device="cpu", enable_memory_saver=False)
        self.alloc = TokenToKVPoolAllocator(size=SIZE, dtype=torch.float16, device="cpu", kvcache=self.pool,
                                            need_sort=False)
        self.r2tp = ReqToTokenPool(size=8, max_context_len=128, device="cpu", enable_memory_saver=False)
        self.reqs, self.tree = ops(self)
        led = K.CardKvLedger(os.path.join(tempfile.mkdtemp(prefix="dcmp"), "card"), "D")
        geom = P._geom_for(torch.zeros(SIZE + 1, 64), SIZE, 1, "k", (SIZE + 1) * 64)
        self.actor = D.DKvStage([(1, geom)], led, allocator=self.alloc, pools=[self.pool], page_size=1, granule=G,
                                top_tokens=SIZE, spans=_Spans(), step=STEP)
        b = self.actor.bytes_for(MAPPED) - self.actor.bytes_for(0)
        led.contribute(b, committed=b)
        self.actor.mapped_tokens, self.actor._committed = MAPPED, b
        P.engage_cap(self.alloc, MAPPED, 1)
        self.sched = types.SimpleNamespace(
            running_batch=types.SimpleNamespace(reqs=self.reqs), waiting_queue=[], chunked_req=None,
            req_to_token_pool=self.r2tp, tree_cache=self.tree, draft_worker=None, tp_rank=r)
        self.fill()

    def owned(self, gid):
        if self.bounds is None:
            return True, gid
        S, lo, hi = self.bounds
        off = gid % S
        return lo <= off < hi, (gid // S) * (hi - lo) + (off - lo)

    def fill(self):
        for gid in range(1, SIZE + 1):
            own, row = self.owned(gid)
            if own:
                for li in range(2):
                    self.pool.k_buffer[li][row] = _val(gid, li, 0)
                    self.pool.v_buffer[li][row] = _val(gid, li, 1)

    def read(self, gid):
        own, row = self.owned(gid)
        if not own:
            return None
        return [float(self.pool.k_buffer[li][row, 0, 0]) for li in range(2)] + \
               [float(self.pool.v_buffer[li][row, 0, 0]) for li in range(2)]


def _seats(rank, *, seat_tokens=(12, 9), filler=203, reserve=3, tree_prefix=0):
    """The metal shape: a big request took the low ids (filler), the seats' rows land above it, the big
    request ends (its rows go back to the TAIL of the free list). ``reserve`` = the spec-v2 rows ahead of
    kv_committed_len (inside kv_allocated_len). ``tree_prefix`` > 0: a locked tree node holds the first
    rows of both seats (the shared system prompt)."""
    a, r2t = rank.alloc, rank.r2tp.req_to_token
    big = a.alloc(filler)
    root = UnifiedTreeNode((ComponentType.FULL,))
    nodes = [root]
    shared = None
    if tree_prefix:
        shared = a.alloc(tree_prefix)
        node = UnifiedTreeNode((ComponentType.FULL,))
        node.parent = root
        node.component_data[ComponentType.FULL].value = shared
        nodes.append(node)
    reqs = []
    for i, n in enumerate(seat_tokens):
        own = a.alloc(n + reserve)
        ids = own if shared is None else torch.cat([shared, own])
        r2t[i, : ids.numel()] = ids.to(r2t.dtype)
        r2t[i, ids.numel(): ids.numel() + 5] = 7          # stale cells beyond the extent (a previous tenant)
        reqs.append(types.SimpleNamespace(rid="weg2-0-%d" % (6 + i), req_pool_idx=i, kv_allocated_len=int(ids.numel()),
                                          kv_committed_len=int(ids.numel()) - reserve,
                                          origin_input_ids=list(range(int(ids.numel()) - reserve)), output_ids=[],
                                          prefix_indices=shared if shared is not None else torch.empty(0, dtype=torch.int64)))
    a.free(big)
    tree = types.SimpleNamespace(root_node=root, _collect_all_nodes=lambda: list(nodes),
                                 tree_components=(ComponentType.FULL, ComponentType.MAMBA),
                                 ongoing_write_through={}, ongoing_load_back={}, ongoing_backup={7: object()},
                                 cache_controller=None)
    return reqs, tree


class Group:
    """A barrier MIN collective over n rank threads (the TP cpu group)."""

    def __init__(self, n):
        self.n, self.bar, self.buf, self.log = n, threading.Barrier(n, timeout=20), [None] * n, []

    def gmin(self, r):
        def g(vals):
            self.buf[r] = [int(v) for v in vals]
            self.bar.wait()
            out = [min(col) for col in zip(*self.buf)]
            if r == 0:
                self.log.append(out)
            self.bar.wait()
            return out
        return g


def _run_group(ranks, target, floor=MAPPED - 1, p_wait_s=5.0):
    grp = Group(len(ranks))
    res, errs = [None] * len(ranks), []

    def body(i, rk):
        try:
            res[i] = C.run(rk.sched, rk.actor, target=target, floor=floor, p_wait_s=p_wait_s, gmin=grp.gmin(i),
                           geometry=(rk.bounds, rk.prefix))
        except Exception as exc:  # noqa: BLE001
            errs.append((i, exc))
            grp.bar.abort()

    ts = [threading.Thread(target=body, args=(i, rk), name="rank-%d" % i) for i, rk in enumerate(ranks)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(30)
    if errs:
        raise errs[0][1]
    return res, grp


def _three(**kw):
    return [Rank(r, (8, PREFIX3[r], PREFIX3[r + 1]), PREFIX3, lambda rk: _seats(rk, **kw)) for r in range(3)]


def _context(rank, req):
    """What attention reads for ``req``: the bytes of every position, from the owner rank's pool."""
    r2t = rank.r2tp.req_to_token
    return [int(x) for x in r2t[req.req_pool_idx, : req.kv_allocated_len]]


def _bytes_by_position(ranks, ri):
    """Position -> the owner rank's bytes of the id the table names there (request ``ri``)."""
    out = []
    for gid in _context(ranks[0], ranks[0].reqs[ri]):
        vals = [rk.read(gid) for rk in ranks]
        owned = [v for v in vals if v is not None]
        assert len(owned) == 1, "exactly one owner per id"
        out.append(owned[0])
    return out


def _snapshot(ranks):
    return [[_bytes_by_position(ranks, i) for i in range(len(ranks[0].reqs))]]


# ------------------------------------------------------------------------------------------- pure

def test_owner_rule_matches_the_hicache_transfer_mapping():
    from sglang.srt.managers.cache_controller import HiCacheController

    ids = torch.arange(0, 300, dtype=torch.int64)
    for r in range(3):
        b = (8, PREFIX3[r], PREFIX3[r + 1])
        fake = types.SimpleNamespace(_dcp_owner_ctx=lambda b=b: b)
        owned_ref, rows_ref = HiCacheController._dcp_owned_device_rows(fake, ids)
        owned, rows = C.compact_rows(ids, b)
        assert torch.equal(owned, owned_ref) and torch.equal(rows[owned], rows_ref)
        assert torch.equal(C.owner_class(ids, PREFIX3) == r, owned)


def test_plan_pairs_lowest_free_of_the_same_owner_class():
    src = torch.tensor([201, 203, 210, 215])
    free = torch.arange(1, 64)
    s, d, per = C.plan_moves(src, free, PREFIX3)
    assert torch.equal(C.owner_class(s, PREFIX3), C.owner_class(d, PREFIX3))
    assert int(d.max()) < 64 and len(set(d.tolist())) == 4
    # class 0 = residues 0..3 (ids 1,2,3,8,..) takes 201/203/210; class 2 = residues 6,7 takes 215 -> 6
    assert d.tolist() == [1, 2, 3, 6]
    with pytest.raises(C.Refused) as e:
        C.plan_moves(torch.tensor([204, 212]), torch.tensor([4, 1, 2]), PREFIX3)   # class 1 has one free id
    assert e.value.reason == "class_short"


def test_gate_dual_d_only_default_on(monkeypatch):
    for env in [{}, {C.SWITCH_ENV: "1"}, {"SGLANG_WEG2_GROUP": "D"},
                {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P", D.MAX_TOKENS_ENV: "256"},
                {**DUAL_D, C.SWITCH_ENV: "0"}, {**DUAL_D, C.SWITCH_ENV: "off"}]:
        assert C.armed(env) is False, env
    assert C.armed(dict(DUAL_D)) is True                  # default ON in the dual D
    assert C.armed({**DUAL_D, C.SWITCH_ENV: "1"}) is True


# ------------------------------------------------------------------------------------------- one rank

def test_single_rank_moves_seats_down_bytes_and_references_follow():
    rk = Rank(0, None, None, lambda r: _seats(r))
    before = [[rk.read(g) for g in _context(rk, q)] for q in rk.reqs]
    avail0 = rk.alloc.available_size()
    target = 96
    assert P.max_live_id(rk.alloc, 1) > target
    res, grp = _run_group([rk], target)
    assert res[0] is not None and res[0] <= target
    after = [[rk.read(g) for g in _context(rk, q)] for q in rk.reqs]
    assert after == before, "every position reads the same bytes after the move"
    assert all(max(_context(rk, q)) <= target for q in rk.reqs)
    assert P.max_live_id(rk.alloc, 1) <= target
    assert rk.alloc.available_size() == avail0, "rows conserved (moved = reserved + freed)"
    # stale cells beyond the extent are not ids of anyone: left alone
    assert int(rk.r2tp.req_to_token[0, rk.reqs[0].kv_allocated_len]) == 7
    # the shrink now goes through on the real stage (cap, CapBreach check)
    rk.actor.group_shrink(target, res[0])            # raises Weg2DualKvCapBreach under a live row
    assert rk.actor.mapped_tokens == target


def test_tree_rows_and_prefix_indices_follow():
    rk = Rank(0, None, None, lambda r: _seats(r, tree_prefix=5))
    node = rk.tree._collect_all_nodes()[1]
    val = node.component_data[ComponentType.FULL].value
    assert int(val.min()) > 96
    before = [[rk.read(g) for g in _context(rk, q)] for q in rk.reqs]
    res, _ = _run_group([rk], 96)
    assert res[0] is not None
    v2 = node.component_data[ComponentType.FULL].value
    assert int(v2.max()) <= 96
    for q in rk.reqs:
        assert torch.equal(q.prefix_indices.to(torch.int64), v2.to(torch.int64))
        assert _context(rk, q)[:5] == v2.tolist(), "table prefix == tree value"
    assert [[rk.read(g) for g in _context(rk, q)] for q in rk.reqs] == before


def test_hicache_in_flight_over_a_tree_row_refuses_nothing_changes():
    rk = Rank(0, None, None, lambda r: _seats(r, tree_prefix=5))
    rk.tree.ongoing_write_through = {1: object()}
    t0 = rk.r2tp.req_to_token.clone()
    f0 = rk.alloc.free_pages.clone()
    res, _ = _run_group([rk], 96)
    assert res[0] is None
    assert torch.equal(rk.r2tp.req_to_token, t0) and torch.equal(rk.alloc.free_pages, f0)


@pytest.mark.parametrize("hold", ["unaccounted", "chunked", "parked", "waiting_rows"])
def test_unknown_holders_refuse_named(hold, caplog):
    rk = Rank(0, None, None, lambda r: _seats(r))
    if hold == "unaccounted":
        rk.reqs.pop()                                     # its rows stay live but nobody known holds them
    elif hold == "chunked":
        rk.sched.chunked_req = types.SimpleNamespace(rid="c")
    elif hold == "parked":
        rk.sched.weg2_d_parked = [object()]
    else:
        rk.sched.waiting_queue = [types.SimpleNamespace(rid="w", req_pool_idx=5)]
    t0 = rk.r2tp.req_to_token.clone()
    import logging

    with caplog.at_level(logging.INFO):
        res, _ = _run_group([rk], 96)
    assert res[0] is None and torch.equal(rk.r2tp.req_to_token, t0)
    assert any("D-COMPACT REFUSED reason=" in r.getMessage() for r in caplog.records)
    assert rk.actor._dc_next_wait_s > 5.0, "backoff on P's group wait"


# ------------------------------------------------------------------------------------------- three ranks, uneven DCP

def test_three_ranks_uneven_dcp_every_rank_identical_and_bytes_kept():
    ranks = _three()
    snap = _snapshot(ranks)
    res, grp = _run_group(ranks, 96)
    assert res[0] is not None and len(set(res)) == 1, "one group floor"
    assert _snapshot(ranks) == snap, "attention reads the same bytes on the owner rank of every position"
    tables = [rk.r2tp.req_to_token.clone() for rk in ranks]
    assert all(torch.equal(tables[0], t) for t in tables[1:]), "replicated table stays replicated"
    frees = [torch.sort(torch.cat([rk.alloc.free_pages, rk.alloc.release_pages])).values for rk in ranks]
    assert all(torch.equal(frees[0], f) for f in frees[1:])
    for rk in ranks:
        assert P.max_live_id(rk.alloc, 1) <= 96
        rk.actor.group_shrink(96, res[0])
        assert rk.actor.mapped_tokens == 96


@pytest.mark.parametrize("how", ["other_rank", "plan_mismatch"])
def test_three_ranks_one_rank_differs_all_refuse_nothing_moves(how, caplog):
    import logging

    ranks = _three()
    if how == "other_rank":
        ranks[2].alloc.alloc(1)                       # rank 2 holds one more high id: unaccounted there
    else:
        fp = ranks[2].alloc.free_pages                # rank 2 lacks the lowest free id: another pairing
        ranks[2].alloc.free_pages = fp[fp != int(fp[(fp >= 1) & (fp <= 96)].min())]
    t0 = [rk.r2tp.req_to_token.clone() for rk in ranks]
    snap = _snapshot(ranks)
    with caplog.at_level(logging.INFO):
        res, grp = _run_group(ranks, 96)
    assert res == [None, None, None], "every rank refuses together"
    assert len(grp.log) == 1, "only the plan collective ran"
    assert all(torch.equal(rk.r2tp.req_to_token, t) for rk, t in zip(ranks, t0))
    assert _snapshot(ranks) == snap
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "reason=" + ("unaccounted" if how == "other_rank" else "plan_mismatch") in msgs


def test_copy_failure_on_one_rank_rolls_back_every_rank():
    ranks = _three()
    t0 = [rk.r2tp.req_to_token.clone() for rk in ranks]
    f0 = [(rk.alloc.free_pages.clone(), rk.alloc.release_pages.clone()) for rk in ranks]
    snap = _snapshot(ranks)
    real = C._copy

    def boom(plan, chunk=C.CHUNK_ROWS):
        n = real(plan, chunk)                         # the bytes were written to dst ...
        if threading.current_thread().name == "rank-1":
            raise RuntimeError("copy failed")         # ... and one rank fails after that
        return n

    with mock.patch.object(C, "_copy", boom):
        res, _ = _run_group(ranks, 96)
    assert res == [None, None, None]
    for rk, t, (fp, rp) in zip(ranks, t0, f0):
        assert torch.equal(rk.r2tp.req_to_token, t)
        assert torch.equal(rk.alloc.free_pages, fp) and torch.equal(rk.alloc.release_pages, rp)
    assert _snapshot(ranks) == snap


# ------------------------------------------------------------------------------------------- draft rows

def _with_mapper(rk):
    m = DraftKVSlotMapper(num_global_slots=SIZE, num_draft_slots=64, ctx_cap=32, device="cpu")
    rk.alloc.register_free_listener(m.on_global_free, m.on_global_clear)
    rk.alloc.register_alias_listener(m.on_global_alias)
    ids = torch.tensor(_context(rk, rk.reqs[0]) + _context(rk, rk.reqs[1]), dtype=torch.int64)
    slots = m.translate_write(ids)
    rk.sched.draft_worker = types.SimpleNamespace(
        draft_model_runner=types.SimpleNamespace(token_to_kv_pool=types.SimpleNamespace(weg2_slot_mapper=m)))
    return m, dict(zip(ids.tolist(), slots.tolist()))


def test_draft_rows_follow_through_the_alias_carry():
    rk = Rank(0, None, None, lambda r: _seats(r))
    m, slot_of = _with_mapper(rk)
    old = {q.rid: _context(rk, q) for q in rk.reqs}
    res, _ = _run_group([rk], 96)
    assert res[0] is not None
    m._drain_pending()
    for q in rk.reqs:
        for g_old, g_new in zip(old[q.rid], _context(rk, q)):
            assert int(m.map[g_new]) == slot_of[g_old], "the draft row of each position moved with it"
            if g_new != g_old:
                assert int(m.map[g_old]) == -1


def test_raw_draft_pool_is_copied_on_every_rank():
    rk = Rank(0, None, None, lambda r: _seats(r))
    draft = MHATokenToKVPool(size=SIZE, page_size=1, dtype=torch.float16, head_num=1, head_dim=2, layer_num=1,
                             device="cpu", enable_memory_saver=False)
    for gid in range(1, SIZE + 1):
        draft.k_buffer[0][gid] = float(gid)
        draft.v_buffer[0][gid] = float(gid) + 0.5
    rk.sched.draft_worker = types.SimpleNamespace(draft_model_runner=types.SimpleNamespace(token_to_kv_pool=draft))
    old = {q.rid: _context(rk, q) for q in rk.reqs}
    res, _ = _run_group([rk], 96)
    assert res[0] is not None
    for q in rk.reqs:
        for g_old, g_new in zip(old[q.rid], _context(rk, q)):
            assert float(draft.k_buffer[0][g_new, 0, 0]) == float(g_old)


# ------------------------------------------------------------------------------------------- the tick

def _tick_rank(monkeypatch, switch="1"):
    _dual(monkeypatch, **{C.SWITCH_ENV: switch, "SGLANG_WEG2_DUAL_D_LIVE_YIELD_WAIT_S": "4"})
    rk = Rank(0, None, None, lambda r: _seats(r))
    led_p = K.CardKvLedger(rk.actor.ledger.path, "P")
    led_p.contribute(0)
    led_p.request(10 ** 9)                                # P waits for a card (demand > 0)
    rk.sched.tp_worker = types.SimpleNamespace(model_runner=types.SimpleNamespace(dual_d_kv=rk.actor))
    rk.sched.server_args = types.SimpleNamespace(chunked_prefill_size=16, speculative_num_draft_tokens=2)
    rk.sched._weg2_group_min_ints = lambda v: list(v)
    rk.tree.evictable_size = lambda: 0
    return rk


def _ticks(rk, n):
    clock = [1000.0]

    def now():
        clock[0] += 1.0
        return clock[0]

    out = []
    with mock.patch.object(D._pk, "_now", now), mock.patch.object(D, "_instr_d_want", lambda *a, **k: None), \
            mock.patch.object(D, "publish_d_signal", lambda *a, **k: None), \
            mock.patch.object(D._pk, "phys_check", lambda *a, **k: None):
        for _ in range(n):
            out.append(D.tick(rk.sched))
    return out


def test_tick_compacts_and_shrinks_in_the_same_tick(monkeypatch, caplog):
    import logging

    rk = _tick_rank(monkeypatch)
    before = [[rk.read(g) for g in _context(rk, q)] for q in rk.reqs]
    with caplog.at_level(logging.INFO):
        verdicts = _ticks(rk, 8)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("D-COMPACT DONE" in m for m in msgs), msgs[-5:]
    assert "shrink" in verdicts and rk.actor.mapped_tokens < MAPPED
    assert rk.actor.mapped_tokens >= P.max_live_id(rk.alloc, 1)
    assert [[rk.read(g) for g in _context(rk, q)] for q in rk.reqs] == before
    done = next(i for i, m in enumerate(msgs) if "D-COMPACT DONE" in m)
    assert any("D-KV SHRINK" in m for m in msgs[done:]), "the shrink follows the compaction"


def test_tick_switch_off_is_the_old_trajectory(monkeypatch):
    rk = _tick_rank(monkeypatch, switch="0")
    ref = _tick_rank(monkeypatch, switch="0")
    with mock.patch.object(D, "_compact_step", lambda sched, actor, **kw: kw["floor"]):
        want = _ticks(ref, 8)
    got = _ticks(rk, 8)
    assert got == want and rk.actor.mapped_tokens == ref.actor.mapped_tokens == MAPPED
    assert torch.equal(rk.r2tp.req_to_token, ref.r2tp.req_to_token)


# ------------------------------------------------------------------------------------------- mutants

def _single_world_check():
    rk = Rank(0, None, None, lambda r: _seats(r))
    before = [[rk.read(g) for g in _context(rk, q)] for q in rk.reqs]
    res, _ = _run_group([rk], 96)
    assert res[0] is not None and res[0] <= 96
    assert all(max(_context(rk, q)) <= 96 for q in rk.reqs), "the table names the low rows"
    assert [[rk.read(g) for g in _context(rk, q)] for q in rk.reqs] == before
    assert P.max_live_id(rk.alloc, 1) <= 96
    rk.actor.group_shrink(96, res[0])
    assert rk.actor.mapped_tokens == 96


def _three_check():
    ranks = _three()
    snap = _snapshot(ranks)
    res, _ = _run_group(ranks, 96)
    assert res[0] is not None
    assert _snapshot(ranks) == snap


def test_mutant_wrong_target_row_turns_red(monkeypatch):
    _single_world_check()
    real = C.compact_rows
    monkeypatch.setattr(C, "compact_rows", lambda ids, b: (lambda o_r: (o_r[0], o_r[1] + 1))(real(ids, b)))
    with pytest.raises(AssertionError):
        _single_world_check()


def test_mutant_owner_class_ignored_under_dcp_turns_red(monkeypatch):
    _three_check()
    real = C.plan_moves
    monkeypatch.setattr(C, "plan_moves", lambda src, free, prefix: real(src, free, None))
    with pytest.raises(AssertionError):
        _three_check()


def test_mutant_references_not_switched_turns_red(monkeypatch):
    monkeypatch.setattr(C, "_commit", lambda sched, plan: None)
    with pytest.raises(AssertionError):
        _single_world_check()


def test_mutant_floor_not_lowered_turns_red(monkeypatch):
    monkeypatch.setattr(C, "_release_src", lambda alloc, plan: None)    # old ids never freed: floor stays
    with pytest.raises(AssertionError):
        _single_world_check()


def test_mutant_one_rank_skips_the_copy_turns_red(monkeypatch):
    _three_check()
    real = C._copy
    monkeypatch.setattr(C, "_copy", lambda plan, chunk=C.CHUNK_ROWS:
                        0 if threading.current_thread().name == "rank-1" else real(plan, chunk))
    with pytest.raises(AssertionError):
        _three_check()


def test_mutant_draft_rows_forgotten_turns_red(monkeypatch):
    def check():
        rk = Rank(0, None, None, lambda r: _seats(r))
        m, slot_of = _with_mapper(rk)
        old = {q.rid: _context(rk, q) for q in rk.reqs}
        _run_group([rk], 96)
        m._drain_pending()
        for q in rk.reqs:
            for g_old, g_new in zip(old[q.rid], _context(rk, q)):
                assert int(m.map[g_new]) == slot_of[g_old]

    check()
    real = C._release_src

    def no_alias(alloc, plan):
        plan.draft = "holes"
        return real(alloc, plan)
    monkeypatch.setattr(C, "_release_src", no_alias)
    with pytest.raises(AssertionError):
        check()


def test_mutant_gate_missing_reaches_the_flip_form(monkeypatch):
    """Flip form (no dual env): the D tick never runs the compaction. Mutant: the gate answers True ->
    with an actor attached outside the dual layout the step would move rows -- the test sees it."""
    rk = Rank(0, None, None, lambda r: _seats(r))
    assert C.armed() is False
    floor = D._compact_step(rk.sched, rk.actor, want=96, floor=MAPPED - 1, live_due=True, p_waiting=True,
                            p_wait_s=9.0, avail_min=150, air=4, holds=False, p_missing=False, recent_grow=False,
                            group_demand=10)
    assert floor == MAPPED - 1
    assert D.tick(types.SimpleNamespace(tp_worker=None)) is None
    monkeypatch.setattr(C, "armed", lambda env=None: True)
    with mock.patch.object(C, "run", lambda *a, **k: 0) as _r:
        got = D._compact_step(rk.sched, rk.actor, want=96, floor=MAPPED - 1, live_due=True, p_waiting=True,
                              p_wait_s=9.0, avail_min=150, air=4, holds=False, p_missing=False, recent_grow=False,
                              group_demand=10)
    assert got != MAPPED - 1, "the mutant gate lets the flip form through -- which is what the first half guards"
