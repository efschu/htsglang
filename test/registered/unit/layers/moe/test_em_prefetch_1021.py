"""EM-PREFETCH (21.10., NF D prefill): an EAGER-DEVICE-EM forward prefetches layer L+1's predicted expert rows on the
pool prefetch side stream, issued right after layer L's own last row copy.

rc30 D (boot ``..._1d822bd150_1021_063252``, 06:36-08:13Z, 788 EM forwards): the EM row copies of the critical rank
are 296.7 of 674.1 s EM time (44 %), all of them serial on the forward stream before each wave's apply; the
wave-1 copies (what a prediction can move ahead) are 207.0 s of it. The pool already carries the decode-graph
prefetch (Mixtral-Offloading's predictor, side stream, joined events) -- capture-only, decode-shaped ids, and
"a prediction is not a use". What must hold here, on the S51 CPU pool harness:

* the prediction becomes the EM pass's one-lane-per-expert form; foreign / pad ids are dropped;
* predicted experts already resident are stamped (protected from the prefetch's own victims), nothing else is:
  the danger direction is a predicted miss evicting a predicted hit (shown with the stamp switched off);
* the prefetch never evicts the last real step's rows, never moves the clock, marks what it filled;
* a landed prefetch turns synchronous misses into hits and changes NO output bit; a wrong one costs rows only;
* L's EM pass fires L+1's prefetch after its LAST row copy and before its last apply;
* the band is decided from boot-wide values (every rank alike), the guards from the layer's own writers;
* every eager path joins a prefetch in flight first; a prediction no EM pass issued is issued after the path;
* on CUDA: begin on the forward stream -> side waits -> body -> done on the side stream, join owed even if the
  body raises; a defect is logged once and switches the prefetch off; the switch off = no prefetch at all;
* the EM-FETCH line carries pf_layers / pf_predicted / pf_fetched / pf_hits / pf_copy_ms on the same one D2H;
* the MoE block predicts with the next block's router on its own MoE input (eager only, inside the band).
"""

import importlib.util
import inspect
import logging
import os
import re
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.layers.moe import em_fetch_probe as P  # noqa: E402
from flliper.srt.layers.moe import expert_offload as eo  # noqa: E402
from flliper.srt.layers.moe import expert_pool_device as ep  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(__file__)

_HERE = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location(
    "_t_em_prefetch_probe", os.path.join(_HERE, "test_s78_em_fetch_probe_1019.py"))
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)
S51 = T.S51
_probe = T._probe  # noqa: F401 -- the autouse fixture (fake events, CPU allowed)

E, R = S51.E, S51.R


@pytest.fixture(autouse=True)
def _switch():
    eo.pool_em_prefetch_reset_for_tests()
    yield
    eo.pool_em_prefetch_reset_for_tests()


def _cold(routes):
    return sorted({e for row in routes for e in row if e >= R})


def _set_use(t, expert, use):
    r = int(t.hot_phys[expert])
    assert r >= t.lru_start, (expert, r)
    t.row_use[r] = use


# ---------------------------------------------------------------------------------------------- lanes, stamp

def test_lanes_are_one_per_predicted_local_expert():
    ids = torch.tensor([[5, 7, -1], [5, 23, 30], [-1, -1, 7]], dtype=torch.int32)
    lanes, mark = eo.em_prefetch_lanes(ids, E)
    assert lanes.shape == (E,)
    assert [int(e) for e in lanes if int(e) >= 0] == [5, 7, 23]
    assert all(int(lanes[e]) in (e, -1) for e in range(E))
    assert int(mark.sum()) == 3  # 30 is no local expert, -1 is padding


def test_stamp_protects_predicted_resident_lru_rows_only(monkeypatch):
    cache = S51._pool_cache(monkeypatch, warm=[8, 9, 10])
    t = cache._pool_tables
    t.clock[0] = 5
    for e in (8, 9, 10):
        _set_use(t, e, 2)
    before = t.row_use.clone()
    lanes, _ = eo.em_prefetch_lanes(torch.tensor([8, 1, 15], dtype=torch.int32), E)
    eo.stamp_predicted_rows(t, lanes)
    assert int(t.row_use[int(t.hot_phys[8])]) == 5        # predicted, resident in the LRU: stamped
    for e in (9, 10):
        assert int(t.row_use[int(t.hot_phys[e])]) == 2    # not predicted: untouched
    changed = (t.row_use != before).nonzero().reshape(-1).tolist()
    assert changed == [int(t.hot_phys[8])]                # a fixed resident (1) and a cold expert (15) touch nothing
    assert int(t.clock[0]) == 5


def test_stamp_never_writes_a_staging_row(monkeypatch):
    cache = S51._pool_cache(monkeypatch)
    t = cache._pool_tables
    staging = int(t.staging_rows[0])
    assert staging >= t.pool_rows
    t.hot_phys[11] = staging  # a staged expert of a step in flight: never an LRU row
    before = t.row_use.clone()
    lanes, _ = eo.em_prefetch_lanes(torch.tensor([11], dtype=torch.int32), E)
    eo.stamp_predicted_rows(t, lanes)
    assert torch.equal(t.row_use, before)


# ---------------------------------------------------------------------------------------------- the prefetch step

def _full_lru(monkeypatch):
    """LRU full of experts 4..12; 6, 7, 8 the oldest, 4 and 5 the last real step's rows (use == clock)."""
    cache = S51._pool_cache(monkeypatch, warm=list(range(4, 13)))
    t = cache._pool_tables
    t.clock[0] = 3
    for e in range(4, 13):
        _set_use(t, e, 1)
    for e in (6, 7, 8):
        _set_use(t, e, 0)
    for e in (4, 5):
        _set_use(t, e, 3)
    return cache, t


def test_prefetch_keeps_hits_and_the_last_step_and_takes_the_coldest_rest(monkeypatch):
    cache, t = _full_lru(monkeypatch)
    pred = torch.tensor([[6, 7, 8], [20, 21, -1]], dtype=torch.int32)
    assert cache.prefetch_pool_em(pred) is True
    bank = cache._resident["w13"]
    for e in (4, 5, 6, 7, 8, 20, 21):
        r = int(t.hot_phys[e])
        assert t.lru_start <= r < t.pool_rows, (e, r)
        assert float(bank[r, 0]) == float(e)              # the bytes are the expert's
    gone = [e for e in (9, 10, 11, 12) if int(t.hot_phys[e]) < 0]
    assert len(gone) == 2                                 # the two victims are unpredicted, unprotected rows
    assert int(t.clock[0]) == 3                           # a prediction never moves the clock
    for e in (20, 21):
        assert int(t.pf_row[int(t.hot_phys[e])]) == 3     # marked: the next real step counts its hits
    assert ep.bijection_breaks(t) == 0 and int(t.error[0]) == 0
    assert cache._em_pf_landed is True


def test_without_the_stamp_a_predicted_miss_evicts_a_predicted_hit(monkeypatch):
    """The danger direction, measured: the decode rule alone trades a row L+1 needs for another."""
    monkeypatch.setattr(eo, "stamp_predicted_rows", lambda tables, lanes: None)
    cache, t = _full_lru(monkeypatch)
    cache.prefetch_pool_em(torch.tensor([[6, 7, 8], [20, 21, -1]], dtype=torch.int32))
    assert [e for e in (6, 7, 8) if int(t.hot_phys[e]) < 0], "the mutant must lose a predicted hit"


def test_no_victim_means_dropped_never_staged(monkeypatch):
    cache = S51._pool_cache(monkeypatch)
    t = cache._pool_tables
    cache.prefetch_pool_em(torch.tensor([S51.WIDE], dtype=torch.int32).reshape(-1))
    lru = t.pool_rows - t.lru_start
    cold = len(_cold(S51.WIDE))
    pred, fetched, hits, skipped = (int(x) for x in t.pf_counts)
    assert fetched == lru and skipped == cold - lru       # 9 free rows filled, the rest dropped
    assert int(cache._pool_pf_buffers.staged_count[0]) == 0


# ---------------------------------------------------------------------------------------------- landing

def test_a_landed_prefetch_turns_misses_into_hits_and_changes_no_bit(monkeypatch):
    on = S51._pool_cache(monkeypatch)
    off = S51._pool_cache(monkeypatch)
    on._pool_tables.misses_total.zero_()
    off._pool_tables.misses_total.zero_()
    on.prefetch_pool_em(torch.tensor(S51.ROUTES, dtype=torch.int32))
    got_on = S51._extend(on, seed=4)
    got_off = S51._extend(off, seed=4)
    assert torch.equal(got_on, got_off)
    assert torch.equal(got_on[:, 0], S51._truth(seed=4))
    t = on._pool_tables
    pred, fetched, hits, skipped = (int(x) for x in t.pf_counts)
    cold = len(_cold(S51.ROUTES))
    assert fetched == hits == 9 and skipped == cold - 9
    assert int(off._pool_tables.misses_total[0]) == cold
    assert int(t.misses_total[0]) == cold - fetched       # the synchronous copies shrank by the hits
    assert ep.bijection_breaks(t) == 0


def test_a_wrong_prediction_costs_rows_never_output(monkeypatch):
    on = S51._pool_cache(monkeypatch)
    wrong = [e for e in range(R, E) if e not in _cold(S51.ROUTES)]
    on.prefetch_pool_em(torch.tensor(wrong, dtype=torch.int32))
    got = S51._extend(on, seed=6)
    assert torch.equal(got[:, 0], S51._truth(seed=6))
    t = on._pool_tables
    assert int(t.pf_counts[2]) == 0                       # no prefetched row was used
    assert int(t.error[0]) == 0 and ep.bijection_breaks(t) == 0


# ---------------------------------------------------------------------------------------------- the chain

def _pair(monkeypatch):
    a = S51._pool_cache(monkeypatch)
    b = S51._pool_cache(monkeypatch)
    b.layer.layer_id = 8
    return a, b


def test_the_em_pass_fires_after_its_last_copy_and_before_its_last_apply(monkeypatch):
    a, b = _pair(monkeypatch)
    calls, copies, seen = [], [0], []
    real_copy = ep.copy_rows

    def _copy(*args):
        copies[0] += 1
        return real_copy(*args)

    monkeypatch.setattr(ep, "copy_rows", _copy)
    real_pf = b.prefetch_pool_em

    def _pf(local_ids, probe=None):
        seen.append((len(calls), copies[0]))
        return real_pf(local_ids, probe=probe)

    monkeypatch.setattr(b, "prefetch_pool_em", _pf)
    a.set_em_prefetch(b, torch.tensor(S51.ROUTES, dtype=torch.int32))
    S51._extend(a, calls=calls)
    assert a.eager_device_em_waves(18)[0] == 2
    # two waves: wave 1 copy + apply, wave 2 copy -> FIRE -> wave 2 apply
    assert seen == [(1, 2)], seen
    assert len(calls) == 2
    assert a._em_pf_pending is None
    assert b._em_pf_landed is True


def test_two_layers_one_forward_land_on_the_em_fetch_line(monkeypatch, caplog):
    a, b = _pair(monkeypatch)
    a.set_em_prefetch(b, torch.tensor(S51.ROUTES, dtype=torch.int32))
    P.on_forward(61)
    with caplog.at_level(logging.INFO, logger=P.__name__):
        S51._extend(a, seed=1)
        got = S51._extend(b, seed=2)
        P.on_forward(62)
    assert torch.equal(got[:, 0], S51._truth(seed=2))
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith(P.MARK)]
    assert len(lines) == 1, lines
    line = lines[0]
    assert "fwd=61 T=6 layers=2 " in line
    m = re.search(r"pf_layers=(\d+) pf_predicted=(\d+) pf_fetched=(\d+) pf_hits=(\d+) pf_copy_ms=([\d.]+)", line)
    assert m, line
    layers, pred, fetched, hits, ms = int(m[1]), int(m[2]), int(m[3]), int(m[4]), float(m[5])
    distinct = len({e for row in S51.ROUTES for e in row})
    assert (layers, pred, fetched, hits) == (1, distinct, 9, 9)
    assert ms > 0
    # the synchronous rows on the line: a's cold misses + b's left over (the ones no free row took)
    cold = len(_cold(S51.ROUTES))
    assert " rows=%d " % (cold + (cold - 9)) in line


def test_a_forward_without_prefetch_keeps_the_old_line(monkeypatch, caplog):
    a = S51._pool_cache(monkeypatch)
    P.on_forward(71)
    with caplog.at_level(logging.INFO, logger=P.__name__):
        S51._extend(a)
        P.on_forward(72)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith(P.MARK)]
    assert len(lines) == 1 and "pf_layers" not in lines[0]


# ---------------------------------------------------------------------------------------------- band and guards

def test_the_band_is_decided_from_boot_wide_values_alike_on_every_rank(monkeypatch):
    small = S51._pool_cache(monkeypatch, scratch=12, staging=3)
    big = S51._pool_cache(monkeypatch, scratch=16, staging=3)  # another rank's capacity
    for n_ids, want in ((18, (True, "em")), (6, (False, "graph_form")), (5, (False, "graph_form"))):
        assert small.em_prefetch_wanted(n_ids) == want == big.em_prefetch_wanted(n_ids)
    with envs.FLLIPER_OPT_MOE_POOL_EAGER_DEVICE_EM_MAX_IDS.override(12):
        assert small.em_prefetch_wanted(18) == (False, "em_max_ids") == big.em_prefetch_wanted(18)


def test_the_guards_name_their_reason(monkeypatch):
    c = S51._pool_cache(monkeypatch)
    assert S51._pool_cache(monkeypatch, width=16).em_prefetch_wanted(18) == (False, "em_width")
    c._rw_plan = object()
    assert c.em_prefetch_wanted(18) == (False, "resume_warm")
    c._rw_plan = None
    c._deferred_rows = object()
    assert c.em_prefetch_wanted(18) == (False, "deferred_rows")
    c._deferred_rows = None
    c._pool_eager_device_em = False
    assert c.em_prefetch_wanted(18) == (False, "em_switch_off")


def test_switch_off_is_no_prefetch_at_all(monkeypatch):
    with envs.FLLIPER_MOE_POOL_EM_PREFETCH.override(False):
        eo.pool_em_prefetch_reset_for_tests()
        a, b = _pair(monkeypatch)
        assert b.em_prefetch_wanted(18) == (False, "switch_off")
        a.set_em_prefetch(b, torch.tensor(S51.ROUTES, dtype=torch.int32))
        S51._extend(a)
        assert b._pool_pf_buffers is None and b._em_pf_landed is False
        from flliper.srt.models import qwen2_moe as q

        assert q.link_moe_pool_em_prefetch([SimpleNamespace(mlp=_blk()), SimpleNamespace(mlp=_blk())]) == 0


# ---------------------------------------------------------------------------------------------- join, fallback

class _Stream:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def wait_event(self, ev):
        self.log.append(("wait", self.name, ev.name))


class _Event:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def record(self, stream=None):
        self.log.append(("record", self.name, getattr(stream, "name", None)))


def _fake_cuda(monkeypatch, log):
    main = _Stream("main", log)
    side = _Stream("side", log)
    cur = {"s": main}

    class _Ctx:
        def __init__(self, s):
            self.s = s

        def __enter__(self):
            self.prev, cur["s"] = cur["s"], self.s
            log.append(("enter", self.s.name))

        def __exit__(self, *a):
            cur["s"] = self.prev
            log.append(("exit", self.s.name))

    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: cur["s"])
    monkeypatch.setattr(torch.cuda, "stream", _Ctx)
    monkeypatch.setattr(eo, "pool_prefetch_stream", lambda: side)
    return main, side, cur


def test_on_cuda_the_side_stream_waits_for_begin_and_the_join_waits_for_done(monkeypatch):
    log = []
    _fake_cuda(monkeypatch, log)
    b = S51._pool_cache(monkeypatch)
    b._pool_pf_begin, b._pool_pf_done = _Event("begin", log), _Event("done", log)
    monkeypatch.setattr(b, "_em_prefetch_streams", lambda: True)
    monkeypatch.setattr(b, "_em_prefetch_body", lambda lanes, waves, probe=None: log.append(("body",)))
    assert b.prefetch_pool_em(torch.tensor(S51.ROUTES, dtype=torch.int32)) is True
    assert log == [("record", "begin", "main"), ("enter", "side"), ("wait", "side", "begin"), ("body",),
                   ("record", "done", "side"), ("exit", "side")]
    assert b._pool_pf_armed is True
    log.clear()
    b._join_pool_prefetch()
    assert log == [("wait", "main", "done")] and b._pool_pf_armed is False
    b._join_pool_prefetch()
    assert log == [("wait", "main", "done")]  # once


def test_on_cuda_a_raising_body_still_owes_the_join(monkeypatch):
    log = []
    _fake_cuda(monkeypatch, log)
    b = S51._pool_cache(monkeypatch)
    b._pool_pf_begin, b._pool_pf_done = _Event("begin", log), _Event("done", log)
    monkeypatch.setattr(b, "_em_prefetch_streams", lambda: True)

    def _boom(lanes, waves, probe=None):
        raise RuntimeError("body")

    monkeypatch.setattr(b, "_em_prefetch_body", _boom)
    with pytest.raises(RuntimeError):
        b.prefetch_pool_em(torch.tensor(S51.ROUTES, dtype=torch.int32))
    assert ("record", "done", "side") in log and b._pool_pf_armed is True


def test_every_eager_path_joins_first_and_a_held_prediction_is_issued_after_it(monkeypatch):
    log = []
    _fake_cuda(monkeypatch, log)
    a, b = _pair(monkeypatch)
    a._pool_pf_armed, a._pool_pf_done = True, _Event("done_a", log)
    order = []
    monkeypatch.setattr(a, "_run_eager_pool_path",
                        lambda d, f, n: (order.append(("path", list(log))), "out")[1])
    monkeypatch.setattr(b, "prefetch_pool_em", lambda ids, probe=None: order.append(("pf", int(ids.numel()))) or True)
    a.set_em_prefetch(b, torch.tensor(S51.ROUTES, dtype=torch.int32))
    ids = torch.tensor(S51.ROUTES, dtype=torch.int32)
    topk = S51.StandardTopKOutput(topk_weights=S51._weights(S51.ROUTES), topk_ids=ids, router_logits=None)
    out = a.run_eager_pool(S51.DispatchOutput(torch.zeros(6, S51.W), None, topk), None)
    assert out == "out"
    assert order[0] == ("path", [("wait", "main", "done_a")])   # joined before the path
    assert order[1] == ("pf", 18)                              # the held prediction, issued after it
    assert a._em_pf_pending is None and a._pool_pf_armed is False


def test_a_defect_is_named_once_and_switches_the_prefetch_off(monkeypatch, caplog):
    a, b = _pair(monkeypatch)

    def _boom(ids, probe=None):
        raise ValueError("bad")

    monkeypatch.setattr(b, "prefetch_pool_em", _boom)
    a.set_em_prefetch(b, torch.tensor(S51.ROUTES, dtype=torch.int32))
    with caplog.at_level(logging.ERROR, logger=eo.__name__):
        got = S51._extend(a, seed=8)
    assert torch.equal(got[:, 0], S51._truth(seed=8))
    assert any("EM-PREFETCH DEFECT layer 7 -> 8" in r.getMessage() for r in caplog.records)
    assert eo.pool_em_prefetch_enabled() is False
    assert b.em_prefetch_wanted(18) == (False, "switch_off")


# ---------------------------------------------------------------------------------------------- the MoE block

def _blk(**kw):
    return SimpleNamespace(pool_em_prefetch_next=None, gate=None, experts=None, topk=None, **kw)


class _Target:
    def __init__(self, want=True):
        self.want, self.issued = want, []

    def em_prefetch_wanted(self, n_ids):
        self.n_ids = n_ids
        return (self.want, "em" if self.want else "graph_form")

    def prefetch_pool_em(self, local, probe=None):
        self.issued.append(local)
        return True


class _Own:
    def __init__(self):
        self.held = []

    def set_em_prefetch(self, target, local):
        self.held.append((target, local))


def _block(monkeypatch, *, want=True, own=True, grouped=False, capture=False):
    from flliper.srt.models import qwen2_moe as q

    monkeypatch.setattr(q, "get_is_capture_mode", lambda: capture)
    g = torch.Generator().manual_seed(3)
    wg = torch.randn(16, E, generator=g)
    gate_calls = []

    def _gate(h):
        gate_calls.append(tuple(h.shape))
        return h @ wg, None

    target = _Target(want)
    nxt = SimpleNamespace(
        experts=SimpleNamespace(_expert_offload=target, _moe_offload_pool=True,
                                pool_prefetch_local_ids=lambda ids: torch.where(ids >= 20, -1, ids)),
        topk=SimpleNamespace(topk_config=SimpleNamespace(use_grouped_topk=grouped, correction_bias=None, top_k=3,
                                                         num_fused_shared_experts=0)),
        gate=_gate,
    )
    own_cache = _Own()
    me = SimpleNamespace(pool_em_prefetch_next=nxt, layer_id=3, _pool_em_prefetch_refused=False,
                         experts=SimpleNamespace(_expert_offload=own_cache if own else None,
                                                 _moe_offload_pool=own))
    return q, me, target, own_cache, gate_calls, wg


def test_the_block_predicts_with_the_next_router_on_its_own_moe_input(monkeypatch):
    q, me, target, own, gate_calls, wg = _block(monkeypatch)
    h = torch.randn(5, 16, generator=torch.Generator().manual_seed(9))
    q.Qwen2MoeSparseMoeBlock._pool_em_prefetch_next_block(me, h)
    assert target.n_ids == 5 * 3
    assert gate_calls == [(5, 16)]
    (tgt, local), = own.held
    assert tgt is target
    want = torch.topk(h @ wg, 3, dim=-1).indices.to(torch.int32)
    assert torch.equal(local, torch.where(want >= 20, -1, want))   # this rank's local ids, foreign = -1
    assert target.issued == []                                     # issued by the pool pass, not here


def test_the_block_stays_silent_under_capture_outside_the_band_and_on_grouped_topk(monkeypatch, caplog):
    h = torch.randn(4, 16)
    q, me, target, own, gate_calls, _ = _block(monkeypatch, capture=True)
    q.Qwen2MoeSparseMoeBlock._pool_em_prefetch_next_block(me, h)
    assert own.held == [] and gate_calls == []
    q, me, target, own, gate_calls, _ = _block(monkeypatch, want=False)
    q.Qwen2MoeSparseMoeBlock._pool_em_prefetch_next_block(me, h)
    assert own.held == [] and gate_calls == []                     # no GEMM outside the band
    q, me, target, own, gate_calls, _ = _block(monkeypatch, grouped=True)
    with caplog.at_level(logging.WARNING, logger=q.__name__):
        q.Qwen2MoeSparseMoeBlock._pool_em_prefetch_next_block(me, h)
        q.Qwen2MoeSparseMoeBlock._pool_em_prefetch_next_block(me, h)
    assert own.held == [] and gate_calls == []
    assert sum("EM-PREFETCH stays off" in r.getMessage() for r in caplog.records) == 1


def test_a_block_without_its_own_pool_pass_issues_at_once(monkeypatch):
    q, me, target, own, gate_calls, _ = _block(monkeypatch, own=False)
    q.Qwen2MoeSparseMoeBlock._pool_em_prefetch_next_block(me, torch.randn(4, 16))
    assert len(target.issued) == 1 and own.held == []


def test_the_chain_links_neighbours_and_breaks_at_a_missing_block(monkeypatch):
    from flliper.srt.models import qwen2_moe as q

    blocks = [_blk(), _blk(), None, _blk(), _blk()]
    layers = [SimpleNamespace(mlp=b) for b in blocks]
    assert q.link_moe_pool_em_prefetch(layers) == 2
    assert blocks[0].pool_em_prefetch_next is blocks[1]
    assert blocks[1].pool_em_prefetch_next is None
    assert blocks[3].pool_em_prefetch_next is blocks[4]
    assert blocks[4].pool_em_prefetch_next is None


def test_topk_knob(monkeypatch):
    from flliper.srt.models import qwen2_moe as q

    monkeypatch.setitem(q._POOL_EM_PREFETCH_TOPK, "m", None)
    assert q.pool_em_prefetch_topk(10) == 10
    monkeypatch.setitem(q._POOL_EM_PREFETCH_TOPK, "m", 4)
    assert q.pool_em_prefetch_topk(10) == 4


def test_the_wiring_is_in_the_forward_and_the_model(monkeypatch):
    """The two call sites a desk harness cannot drive: the block's forward predicts before its router runs,
    the Qwen3.5 backbone (Qwen4Exp's base) links the chain."""
    from flliper.srt.models import qwen2_moe as q
    from flliper.srt.models import qwen3_5

    fwd = inspect.getsource(q.Qwen2MoeSparseMoeBlock.forward)
    i = fwd.index("self._pool_em_prefetch_next_block(hidden_states)")
    assert i < fwd.index("self._forward_router_experts(hidden_states)")
    assert "link_moe_pool_em_prefetch(self.layers)" in inspect.getsource(qwen3_5)


# ---------------------------------------------------------------------------------------------- review A1-A4, E1, E2

def _tables_snapshot(t):
    return {k: getattr(t, k).clone() for k in ("hot_phys", "row_key", "row_use", "pf_row", "pf_counts", "clock",
                                               "miss_count", "error")}


def test_review_a1_a_raise_after_the_step_takes_its_filled_rows_back(monkeypatch):
    """B-1: the copy raises after the prefetch step wrote the tables -- no table entry may name a row whose bytes
    never arrived; the predicted experts are simply not resident, the next real step fetches them."""
    b = S51._pool_cache(monkeypatch, warm=[8, 9])
    t = b._pool_tables
    real = ep.copy_rows

    def _copy(srcs, dsts, src_rows, dst_rows, count):
        if b._pool_pf_buffers is not None and src_rows is b._pool_pf_buffers.gather_src:
            raise RuntimeError("copy launch failed")
        return real(srcs, dsts, src_rows, dst_rows, count)

    monkeypatch.setattr(ep, "copy_rows", _copy)
    with pytest.raises(RuntimeError):
        b.prefetch_pool_em(torch.tensor(S51.ROUTES, dtype=torch.int32))
    clock = int(t.clock[0])
    lo, hi = t.lru_start, t.pool_rows
    assert not bool((t.pf_row[lo:hi] == clock).any())
    for e in _cold(S51.ROUTES):
        if e in (8, 9):
            continue
        assert int(t.hot_phys[e]) < 0, e                  # never a row without its bytes
    for e in (8, 9):                                      # untouched residents keep their rows and bytes
        r = int(t.hot_phys[e])
        assert r >= lo and float(b._resident["w13"][r, 0]) == float(e)
    assert ep.bijection_breaks(t) == 0 and int(t.error[0]) == 0
    assert b._em_pf_landed is False
    monkeypatch.setattr(ep, "copy_rows", real)
    got = S51._extend(b, seed=12)
    assert torch.equal(got[:, 0], S51._truth(seed=12))


def test_review_a1_on_cuda_a_raising_copy_lands_nothing_and_still_owes_the_join(monkeypatch):
    log = []
    _fake_cuda(monkeypatch, log)
    b = S51._pool_cache(monkeypatch)
    b._pool_pf_begin, b._pool_pf_done = _Event("begin", log), _Event("done", log)
    monkeypatch.setattr(b, "_em_prefetch_streams", lambda: True)
    monkeypatch.setattr(ep, "copy_rows", lambda *a: (_ for _ in ()).throw(RuntimeError("copy")))
    with pytest.raises(RuntimeError):
        b.prefetch_pool_em(torch.tensor(S51.ROUTES, dtype=torch.int32))
    assert ("record", "done", "side") in log
    assert b._pool_pf_armed is True and b._em_pf_landed is False


def test_review_a2_the_block_without_its_own_pool_goes_through_the_defect_bracket(monkeypatch, caplog):
    q, me, target, own, gate_calls, _ = _block(monkeypatch, own=False)

    def _boom(local, probe=None):
        raise ValueError("direct")

    target.prefetch_pool_em = _boom
    with caplog.at_level(logging.ERROR, logger=eo.__name__):
        q.Qwen2MoeSparseMoeBlock._pool_em_prefetch_next_block(me, torch.randn(4, 16))   # does not raise
    assert any("EM-PREFETCH DEFECT layer 3" in r.getMessage() for r in caplog.records)
    assert eo.pool_em_prefetch_enabled() is False


def test_review_a3_promote_deferred_joins_a_prefetch_in_flight(monkeypatch):
    log = []
    _fake_cuda(monkeypatch, log)
    b = S51._pool_cache(monkeypatch)
    b._pool_pf_buffers = ep.allocate_step_buffers("cpu", E, S51.WIDTH)
    b._pool_pf_armed, b._pool_pf_done = True, _Event("done", log)
    b._deferred_rows = SimpleNamespace(full=None)
    b._promote_deferred()
    assert log == [("wait", "main", "done")] and b._pool_pf_armed is False


def test_review_a4_the_join_records_the_visible_wait_and_the_line_carries_it(monkeypatch, caplog):
    log = []
    _fake_cuda(monkeypatch, log)
    b = S51._pool_cache(monkeypatch)
    P.on_forward(81)
    rec = P.begin_layer(torch.device("cpu"))
    end = rec.mark()                                     # the side stream's copy end ...
    b._pool_pf_armed, b._pool_pf_done, b._em_pf_end_ev = True, _Event("done", log), end
    b._join_pool_prefetch()                              # ... reached by the forward stream after it: no wait
    assert len(rec.pf_waits) == 1 and b._em_pf_end_ev is None
    late = rec.mark()
    rec.prefetch_wait(rec.pf_waits[0][0], late)          # a copy that ended 1.5 ms after the join: a wait
    rec.pf_layers = 1
    rec.end_layer(6)
    with caplog.at_level(logging.INFO, logger=P.__name__):
        P.on_forward(82)
    line = [r.getMessage() for r in caplog.records if r.getMessage().startswith(P.MARK)][-1]
    assert "pf_wait_ms=1.5" in line, line


def test_review_e1_the_prefetch_variant_is_built_at_install_as_a_table_no_op(monkeypatch):
    b = S51._pool_cache(monkeypatch, warm=[8, 9])
    calls = []
    real = ep.step
    monkeypatch.setattr(ep, "step", lambda *a, **k: (calls.append(k.get("prefetch")), real(*a, **k))[1])
    before = _tables_snapshot(b._pool_tables)
    assert b.em_prefetch_prebuild() is True
    assert calls == [True]
    assert b._pool_pf_buffers.gather_src.shape[0] == S51.WIDTH   # the EM width the live prefetch uses
    for k, v in _tables_snapshot(b._pool_tables).items():
        assert torch.equal(v, before[k]), k
    assert int(b._pool_pf_buffers.gather_count[0]) == 0
    with envs.FLLIPER_MOE_POOL_EM_PREFETCH.override(False):
        eo.pool_em_prefetch_reset_for_tests()
        assert S51._pool_cache(monkeypatch).em_prefetch_prebuild() is False


def test_review_e2_the_band_follows_the_em_pass_verdict(monkeypatch):
    c = S51._pool_cache(monkeypatch)
    c._route_note = True
    assert c.em_prefetch_wanted(18) == (False, "host_instrument")
    c._route_note = False
    with envs.FLLIPER_OPT_MOE_POOL_EAGER_DEVICE_EM_WAVES.override(1):
        assert c.em_prefetch_wanted(18) == (False, "em_waves_cap")
    assert c.em_prefetch_wanted(18) == (True, "em")
