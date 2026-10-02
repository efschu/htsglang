# SPDX-License-Identifier: Apache-2.0
"""L15-10 S4n-e: a P stage adopts a hot follow-up's prefix at admission
(orchestration on the REAL tree; the copies are covered by the take tests)."""

from __future__ import annotations

import importlib.util
import os

from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker as IC,
)
from sglang.srt.weg2 import l15_share_admit as sa
from sglang.srt.weg2 import l15_share_take

_HERE = os.path.dirname(__file__)


def _fx():
    spec = importlib.util.spec_from_file_location(
        "test_weg2_l15_tree_rewrite_1001",
        os.path.join(_HERE, "test_weg2_l15_tree_rewrite_1001.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, m._fixture()


def _admit(fx, monkeypatch, *, take_raises=None, fetch_raises=False):
    calls = {}

    def fake_kv(shares, *, rid, n, stage_layers, p_buffers, p_rows, map_extent,
                skip_ranks=()):
        if take_raises:
            raise l15_share_take.L15TakeError(take_raises)
        calls["rows"] = list(p_rows)
        return n * len(stage_layers)

    def fake_anchor(shares, *, rid, spec, ratios, stage, p_temporal, p_conv,
                    p_slot, map_extent):
        calls["slot"] = p_slot
        return 64

    monkeypatch.setattr(l15_share_take, "take_kv", fake_kv)
    monkeypatch.setattr(l15_share_take, "take_anchor", fake_anchor)

    def fetch(r):
        if fetch_raises:
            from sglang.srt.weg2.l15_hold_share import L15ShareError
            raise L15ShareError("no hold share published for D rank %d" % r)
        return ({"prefix": [0, 2, 3], "spans": []}, [])

    logs = []
    why = sa.admit(rid="r2", token_ids=list(range(700, 740)),
                   hint={"prev_rid": "r1", "n": 30}, fetch=fetch, n_d_ranks=2,
                   kv_alloc=fx.allocator, mamba_alloc=fx.pool.mamba_allocator,
                   tree_cache=fx.cache, stage_att_layers=[1, 3], p_buffers={},
                   spec=None, stage_linear=(0, 2), p_temporal=None, p_conv=None,
                   map_extent=None, log=logs.append)
    return why, calls, logs


def test_success_adopts_the_prefix_as_device_nodes(monkeypatch):
    m, fx = _fx()
    why, calls, logs = _admit(fx, monkeypatch)
    assert why is None, why
    hit = m._match(fx, list(range(700, 730)))
    assert hit.device_indices.tolist() == calls["rows"][: len(hit.device_indices)]
    assert len(hit.device_indices) >= 29
    dup, _i, shared = IC._mamba_double_claimed(fx.pool.mamba_allocator, fx.cache)
    assert dup == 0 and shared == 0
    assert logs and logs[0].startswith("HOT-HANDOVER rid=r2 from=r1 n=30")


def test_take_refusal_leaves_tree_and_allocators_untouched(monkeypatch):
    m, fx = _fx()
    free_kv = fx.allocator.available_size()
    free_mb = fx.pool.mamba_allocator.available_size()
    why, _c, _l = _admit(fx, monkeypatch, take_raises="rid r1 is not held by D")
    assert why.startswith("take: rid r1 is not held")
    assert fx.allocator.available_size() == free_kv
    assert fx.pool.mamba_allocator.available_size() == free_mb
    assert len(m._match(fx, list(range(700, 730))).device_indices) == 0


def test_missing_share_is_named(monkeypatch):
    m, fx = _fx()
    why, _c, _l = _admit(fx, monkeypatch, fetch_raises=True)
    assert why.startswith("share: no hold share published")


def test_hint_file_roundtrip(tmp_path):
    sa.write_hot_hint(str(tmp_path), "r2", "r1", 30)
    assert sa.hot_hint(str(tmp_path), "r2") == {"prev_rid": "r1", "n": 30}
    sa.reap_hot_hint(str(tmp_path), "r2")
    assert sa.hot_hint(str(tmp_path), "r2") is None


# -- L15-10c: cap-0 tokens + anchor from L2, one verdict for every stage -----

from sglang.srt.weg2.l15_share_publish import _b64  # noqa: E402

PREFIX = [0, 2, 3]       # rank 0 owns slot%3 in {0,1} (cap 0), rank 1 slot%3==2


class _L2:
    def __init__(self, kv_why=None, a_why=None):
        self.kv_calls, self.a_calls = [], []
        self.kv_why, self.a_why = kv_why, a_why

    def kv(self, rows, slots, gens):
        self.kv_calls.append((list(rows), list(slots), list(gens)))
        return self.kv_why

    def anchor(self, slot, gen, p_slot):
        self.a_calls.append((slot, gen, p_slot))
        return self.a_why


def _admit_cap0(fx, monkeypatch, l2, verdict=None, n=30):
    seen = {}

    def fake_kv(shares, *, rid, n, stage_layers, p_buffers, p_rows, map_extent,
                skip_ranks=()):
        seen["skip"] = list(skip_ranks)
        seen["ranks"] = sorted(shares)
        seen["rows"] = list(p_rows)
        return 1

    def no_anchor(*a, **k):
        raise AssertionError("the anchor must come from L2")

    monkeypatch.setattr(l15_share_take, "take_kv", fake_kv)
    monkeypatch.setattr(l15_share_take, "take_anchor", no_anchor)
    slots = list(range(100, 100 + n))
    span = {"rid": "r1", "slots": slots, "anchor_l2_slot": 41, "anchor_l2_gen": 6,
            "l2_slots_b64": _b64([1000 + i for i in range(n)]),
            "l2_gens_b64": _b64([9] * n)}
    fetched = []

    def fetch(r):
        fetched.append(r)
        return ({"prefix": PREFIX, "spans": [span]}, [])

    logs = []
    why = sa.admit(rid="r2", token_ids=list(range(700, 700 + n)),
                   hint={"prev_rid": "r1", "n": n}, fetch=fetch, n_d_ranks=2,
                   kv_alloc=fx.allocator, mamba_alloc=fx.pool.mamba_allocator,
                   tree_cache=fx.cache, stage_att_layers=[1], p_buffers={},
                   spec=None, stage_linear=(0, 2), p_temporal=None, p_conv=None,
                   map_extent=None, log=logs.append, cap0=[0], l2=l2,
                   verdict=verdict)
    return why, seen, fetched, slots, logs


def test_cap0_tokens_and_the_anchor_come_from_l2(monkeypatch):
    m, fx = _fx()
    l2 = _L2()
    why, seen, fetched, slots, logs = _admit_cap0(fx, monkeypatch, l2)
    assert why is None, why
    assert fetched == [1] and seen["skip"] == [0]       # no share from cap-0
    rows, l2s, gens = l2.kv_calls[0]
    owned0 = [i for i, s in enumerate(slots) if s % 3 in (0, 1)]
    assert rows == [seen["rows"][i] for i in owned0]
    assert l2s == [1000 + i for i in owned0] and set(gens) == {9}
    assert l2.a_calls and l2.a_calls[0][:2] == (41, 6)
    assert "l2_tokens=%d anchor=l2 adopted" % len(owned0) in logs[0]


def test_an_l2_refusal_is_named_and_adopts_nothing(monkeypatch):
    m, fx = _fx()
    free_kv = fx.allocator.available_size()
    why, *_ = _admit_cap0(fx, monkeypatch, _L2(kv_why="3 page(s) not COMPLETE"))
    assert why.startswith("take: L2 tokens: 3 page(s) not COMPLETE")
    assert fx.allocator.available_size() == free_kv


def test_a_peer_stage_fallback_verdict_adopts_nothing(monkeypatch):
    m, fx = _fx()
    votes = []
    why, *_ = _admit_cap0(fx, monkeypatch, _L2(),
                          verdict=lambda ok: votes.append(ok) or "fallback")
    assert votes == [True] and "verdict fallback" in why
    assert len(m._match(fx, list(range(700, 730))).device_indices) == 0


def test_stage_verdict_all_ok_one_fail_and_timeout(tmp_path):
    d = str(tmp_path)
    t = [0.0]

    def tick(_s):
        t[0] += 0.01

    # stage 0 posts first and waits; stage 1 completes the set -> adopt
    import threading
    out = {}
    th = threading.Thread(target=lambda: out.setdefault(
        0, sa.stage_verdict(d, "rA", 0, 2, True, 5.0)))
    th.start()
    out[1] = sa.stage_verdict(d, "rA", 1, 2, True, 5.0)
    th.join()
    assert out == {0: "adopt", 1: "adopt"}
    # a failing stage decides fallback at once
    assert sa.stage_verdict(d, "rB", 1, 2, False, 5.0) == "fallback"
    assert sa.stage_verdict(d, "rB", 0, 2, True, 5.0) == "fallback"
    # a missing peer -> fallback after the timeout (fake clock)
    assert sa.stage_verdict(d, "rC", 0, 3, True, 0.05, sleep=tick,
                            now=lambda: t[0]) == "fallback"
    assert sa.reap_verdict(d, "rA") == 3
    assert not [x for x in tmp_path.iterdir() if x.name.startswith("hotv.rA.")]


def test_reap_hint_takes_the_verdict_files_and_stale_files_are_swept(tmp_path):
    d = str(tmp_path)
    sa.write_hot_hint(d, "r9", "r8", 4)
    sa.stage_verdict(d, "r9", 0, 1, True, 1.0)
    sa.reap_hot_hint(d, "r9")
    assert list(tmp_path.iterdir()) == []
    (tmp_path / "hotv.old.verdict").write_text("adopt")
    os.utime(tmp_path / "hotv.old.verdict", (1, 1))
    assert sa.sweep_stale(d, max_age_s=60.0) == 1


def test_l2_loader_checks_generations_then_loads_once():
    from types import SimpleNamespace

    loads = []
    hp = SimpleNamespace(slot_gens=lambda s: [9 if x != 1003 else 8 for x in s],
                         _arena_page_tokens=1,
                         _load_pages_all_layers=lambda dev, sl, di, lanes, mode:
                         loads.append((sl.tolist(), di.tolist())))
    L = sa.L2Loader(hp, object(), None, None)
    assert L.kv([5, 6], [1001, 1002], [9, 9]) is None
    assert loads == [([1001, 1002], [5, 6])]
    assert "not COMPLETE" in L.kv([7], [1003], [9])
    assert len(loads) == 1
    hm_loads = []
    hm = SimpleNamespace(slot_gens=lambda s: [6], _load_states_all_layers=lambda dev, s, d:
                         hm_loads.append((s.tolist(), d.tolist())))
    A = sa.L2Loader(None, None, hm, object())
    assert A.anchor(41, 6, 3) is None and hm_loads == [([41], [3])]
    assert "generation" in A.anchor(41, 7, 3)
    assert "no mamba host" in sa.L2Loader(None, None, None, None).anchor(1, 1, 1)
