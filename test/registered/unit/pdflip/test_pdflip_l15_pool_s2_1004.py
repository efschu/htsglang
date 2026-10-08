# SPDX-License-Identifier: Apache-2.0
"""L15-POOL stage S2 (KV, no anchor): admission against the SUM for the rank
without a home segment, the plan's guest-room check BEFORE the retain, the
group agreement on the pool plan, the park path as the guest transport.

Hermetic: no CUDA, no model, no network.

Run (own worktree, capped):
  cd /spinning/wt-27b-l15-pool-s2-1004 && /spinning/gpu-arb/pytest_gedeckelt.sh \
    test/registered/unit/pdflip/test_pdflip_l15_pool_s2_1004.py

Pinned:
* ``select_hold_pool`` == ``select_hold`` when no rank is cap 0; with a cap-0
  rank it never admits more than the SUM of the capped ranks' free rows holds
  (property, random inputs) and says ``pool_full`` (mutant: guest check gone);
* ``plan_round(pool=True)`` drops requests until ``park_plan`` places every
  guest row on the COMPACTED rows -- before anything moves (mutant: the check
  gone -> a hold without room); ``pool=False`` is today's plan byte for byte;
* the group vote of a pooled round carries the plan digest: ranks on
  different plans turn the round off for everyone, ONE gather (mutant:
  digest comparison gone);
* the park under the pool: the plan digest is part of the vote (no collective
  on diverging plans), the source checksum catches a guest row that came back
  wrong -> group fallback to the L2 refill, POOL-OUT/BACK/CHECK lines; off =
  the sidecar and the lines of the per-card path unchanged;
* the switch: default off, refused by name for dual / non-27B / no master.
"""

from __future__ import annotations

import importlib
import inspect
import random
import sys
import threading
import types

import pytest
import torch

from flliper.srt.environ import envs
from flliper.srt.pdflip import l15_park as P
from flliper.srt.pdflip import l15_plan, l15_pool, l15_retain
from flliper.srt.pdflip import l15_sleep_agree as A
from flliper.srt.pdflip.l15_policy import Candidate, select_hold

import test_pdflip_l15_park_1002 as T  # _World, _bufs, _entries, _run_ranks


def _mutant(modname, old, new, *more):
    """The module with source lines changed (a mutant to be killed);
    ``more`` = further (old, new) pairs."""
    mod = importlib.import_module(modname)
    src = inspect.getsource(mod)
    for o, n in ((old, new),) + tuple(zip(more[0::2], more[1::2])):
        assert src.count(o) == 1, (modname, o, src.count(o))
        src = src.replace(o, n)
    m = types.ModuleType(modname + "_mut")
    m.__file__ = mod.__file__
    sys.modules[m.__name__] = m
    exec(compile(src, mod.__file__, "exec"), m.__dict__)
    return m


def _cand(rid, kind, rows, last=1.0):
    return Candidate(rid=rid, kind=kind, last_active=last, rows_by_rank=tuple(rows),
                     anchor_depth=sum(rows), kv_depth=sum(rows))


# -- select_hold_pool ------------------------------------------------------------


def test_equals_select_hold_when_no_rank_is_cap_zero():
    rng = random.Random(7)
    for _ in range(300):
        n = rng.choice([2, 3, 4])
        caps = [rng.randint(1, 40) for _ in range(n)]
        cands = [_cand("r%d" % i, rng.choice(["seat", "parked", "served"]),
                       [rng.randint(0, 9) for _ in range(n)], rng.random())
                 for i in range(rng.randint(1, 8))]
        cap_a = rng.randint(1, 6)
        a = select_hold(cands, caps, cap_a)
        b = l15_pool.select_hold_pool(cands, caps, cap_a)
        assert a == b


def test_cap0_rank_needs_guest_room_today_waves_it_through():
    # caps (0, 8, 8): 16 free rows; a request of (4,4,4) rows = 12, two = 24.
    cands = [_cand("a", "seat", (4, 4, 4)), _cand("b", "parked", (4, 4, 4))]
    caps = (0, 8, 8)
    today = select_hold(cands, caps, 8)
    assert today.rids == ("a", "b")                      # no room check for cap 0
    pool = l15_pool.select_hold_pool(cands, caps, 8)
    assert pool.rids == ("a",)
    assert ("b", "pool_full") in pool.excluded


def test_capped_rank_over_its_own_cap_is_still_no_room_and_anchor_full_kept():
    cands = [_cand("a", "seat", (0, 9, 0)), _cand("b", "parked", (0, 1, 0))]
    pool = l15_pool.select_hold_pool(cands, (0, 8, 8), 8)
    assert pool.rids == ("b",) and ("a", "no_room") in pool.excluded
    cands = [_cand("a", "seat", (1, 1, 1)), _cand("b", "parked", (1, 1, 1))]
    pool = l15_pool.select_hold_pool(cands, (0, 8, 8), 1)
    assert pool.rids == ("a",) and ("b", "anchor_full") in pool.excluded


def test_anchorless_excluded_and_order_is_todays():
    c_bad = Candidate("x", "seat", 9.0, (1, 1, 1), anchor_depth=3, kv_depth=4)
    cands = [_cand("s", "served", (1, 1, 1), 9.0), _cand("p", "parked", (1, 1, 1), 1.0),
             _cand("t", "seat", (1, 1, 1), 0.0), c_bad]
    pool = l15_pool.select_hold_pool(cands, (0, 8, 8), 8)
    assert pool.rids == ("t", "p", "s") and ("x", "anchorless") in pool.excluded


def _sum_fits(hs, cands, caps):
    by = {c.rid: c for c in cands}
    n = len(caps)
    home = [sum(by[r].rows_by_rank[k] for r in hs.rids) if caps[k] > 0 else 0
            for k in range(n)]
    guest = sum(by[r].rows_by_rank[k] for r in hs.rids for k in range(n) if caps[k] == 0)
    return (all(home[k] <= caps[k] for k in range(n) if caps[k] > 0)
            and guest <= sum(caps[k] - home[k] for k in range(n) if caps[k] > 0))


def _random_cases():
    rng = random.Random(11)
    for _ in range(400):
        n = rng.choice([2, 3, 4])
        caps = [rng.randint(1, 30) for _ in range(n)]
        caps[rng.randrange(n)] = 0
        if rng.random() < 0.3:
            caps[rng.randrange(n)] = 0
        cands = [_cand("r%d" % i, rng.choice(["seat", "parked", "served"]),
                       [rng.randint(0, 9) for _ in range(n)], rng.random())
                 for i in range(rng.randint(1, 9))]
        yield caps, cands


def test_never_admits_more_than_the_sum_holds_property():
    n_checked = 0
    for caps, cands in _random_cases():
        hs = l15_pool.select_hold_pool(cands, caps, 99)
        assert _sum_fits(hs, cands, caps), (caps, hs)
        n_checked += 1
    assert n_checked == 400


def test_mutant_hold_without_room_is_killed_in_the_policy():
    mut = _mutant("flliper.srt.pdflip.l15_pool", "if need > free_guest:", "if False:")
    bad = 0
    for caps, cands in _random_cases():
        hs = mut.select_hold_pool(cands, caps, 99)
        if not _sum_fits(hs, cands, caps):
            bad += 1
    assert bad > 0, "the property does not see a hold admitted without guest room"
    # ... and the named case
    cands = [_cand("a", "seat", (4, 4, 4)), _cand("b", "parked", (4, 4, 4))]
    assert mut.select_hold_pool(cands, (0, 8, 8), 8).rids == ("a", "b")


# -- plan_round ------------------------------------------------------------------

PREFIX = [0, 1, 2, 3]


def _plan_inputs(n_req=3):
    cands, slots = [], {}
    for i in range(n_req):
        rid = "q%d" % i
        cands.append(_cand(rid, ("seat", "parked", "served")[i % 3], (4, 4, 4), 10.0 - i))
        slots[rid] = tuple(range(1 + 12 * i, 13 + 12 * i))
    return cands, slots


def _plan(caps, pool, n_req=3, log=None, mod=l15_retain):
    cands, slots = _plan_inputs(n_req)
    return mod.plan_round(
        candidates=cands, slots_of=lambda rid: slots[rid], anchor_slot_of=lambda rid: 5,
        caps_rows_by_rank=caps, cap_anchor_slots=8, prefix=PREFIX, epoch=3,
        log=log or (lambda s: None), pool=pool)


def test_pool_off_is_todays_plan_byte_for_byte():
    caps = (0, 40, 40)
    a = _plan(caps, False)
    assert a.guest_pieces is None and a.pool_fp is None
    cands, slots = _plan_inputs()
    old = l15_retain.plan_round(
        candidates=cands, slots_of=lambda rid: slots[rid], anchor_slot_of=lambda rid: 5,
        caps_rows_by_rank=caps, cap_anchor_slots=8, prefix=PREFIX, epoch=3,
        log=lambda s: None)                       # no pool argument at all
    assert (old.hs, old.plan, old.a_h, old.new_anchors) == (a.hs, a.plan, a.a_h, a.new_anchors)


def test_pool_plan_places_every_guest_row_before_any_move():
    caps = (0, 40, 40)
    logs = []
    rp = _plan(caps, True, log=logs.append)
    assert rp is not None and rp.guest_pieces is not None and rp.pool_fp
    pieces, why = P.park_plan(list(rp.plan.rows_by_rank), list(caps))
    assert why is None and tuple(pieces) == rp.guest_pieces
    assert sum(p.rows for p in rp.guest_pieces) == rp.plan.rows_by_rank[0]
    assert any(x.startswith("L15-POOL-PLAN at=sleep epoch=3") and "guest_rows=" in x
               and "fp=" + rp.pool_fp in x for x in logs)


@pytest.mark.parametrize("caps", [(0, 14, 14), (0, 10, 6), (0, 8, 8), (0, 24, 0), (0, 5, 5)])
def test_pool_plan_never_holds_without_room(caps):
    rp = _plan(caps, True, n_req=3)
    if rp is None:
        return                                     # nothing fit: benign skip
    rows = list(rp.plan.rows_by_rank)
    assert all(rows[r] <= caps[r] for r in range(3) if caps[r] > 0)
    assert P.park_plan(rows, list(caps))[1] is None
    # the pool decision is a prefix of the priority order (lowest dropped first)
    all_rids = ["q0", "q1", "q2"]
    assert list(rp.hs.rids) == all_rids[:len(rp.hs.rids)]


def test_pool_plan_drops_lowest_priority_with_reason_pool_full():
    # whole requests: 12 rows each; pool 2*16=32 free in the capped ranks, so
    # at most two requests' worth of compacted rows (home + guest) can fit
    caps = (0, 16, 16)
    rp = _plan(caps, True, n_req=3)
    assert rp is not None and 0 < len(rp.hs.rids) < 3
    dropped = [r for r in ("q0", "q1", "q2") if r not in rp.hs.rids]
    ex = dict(rp.hs.excluded)
    assert all(ex[r] in ("pool_full", "keep_over_cap") for r in dropped)


def test_mutant_plan_without_the_guest_check_holds_without_room():
    caps = (0, 12, 12)
    ok = _plan(caps, True, n_req=3)
    if ok is not None:
        assert P.park_plan(list(ok.plan.rows_by_rank), list(caps))[1] is None
    # (1) the trim loop's check gone: the final safety net still refuses the
    # round (None = nothing is held), it never holds without room
    loop_only = _mutant("flliper.srt.pdflip.l15_retain",
                        "            if not pool:\n                return None\n",
                        "            return None\n")
    rp1 = _plan(caps, True, n_req=3, mod=loop_only)
    assert rp1 is None or P.park_plan(list(rp1.plan.rows_by_rank), list(caps))[1] is None
    # (2) both gone: a hold without room appears -- the test sees it
    both = _mutant("flliper.srt.pdflip.l15_retain",
                   "            if not pool:\n                return None\n",
                   "            return None\n",
                   "        if _gwhy is not None:\n",
                   "        if False:\n")
    bad = _plan(caps, True, n_req=3, mod=both)
    assert bad is not None
    assert P.park_plan(list(bad.plan.rows_by_rank), list(caps))[1] is not None, (
        "the mutant did not produce a hold without room -- the test is blind")


# -- the group vote (rank agreement) ------------------------------------------------


class _Group:
    def __init__(self, n):
        self.n = n
        self.bar = threading.Barrier(n, timeout=20)
        self.slots = [None] * n
        self.calls = [0] * n

    def gather_for(self, r):
        def g(v):
            self.calls[r] += 1
            self.slots[r] = v
            self.bar.wait()
            out = list(self.slots)
            self.bar.wait()
            return out
        return g


def _kwargs(caps, n_req=2):
    cands, slots = _plan_inputs(n_req)
    return dict(
        candidates=list(cands), slots_of=lambda rid: slots[rid],
        anchor_slot_of=lambda rid: 5, caps_rows_by_rank=caps, cap_anchor_slots=8,
        prefix=PREFIX, epoch=8, pid=1,
        l2_of=lambda rid: (tuple(range(100, 112)), (1,) * 12),
        anchor_l2_of=lambda rid: (7, 1), l2_lanes_of=lambda rid: ())


def _decide(per_rank, pool, mod=A):
    grp = _Group(len(per_rank))
    out = [None] * len(per_rank)

    def work(r):
        kw, cap = per_rank[r]
        out[r] = mod.decide_first(kw, False, lambda: cap, lambda: (r, PREFIX),
                                  grp.gather_for(r), lambda s: None, lambda s: None,
                                  pool=pool)

    ts = [threading.Thread(target=work, args=(r,)) for r in range(len(per_rank))]
    [t.start() for t in ts]
    [t.join(30) for t in ts]
    assert all(o is not None for o in out), "a rank never returned (hang)"
    return out, grp


def test_pooled_round_agrees_when_every_rank_plans_the_same_pool():
    caps = (0, 40, 40)
    out, grp = _decide([(_kwargs(caps), c) for c in caps], True)
    assert [d for _p, d in out] == [None, None, None]
    fps = {p.pool_fp for p, _d in out}
    assert len(fps) == 1 and None not in fps
    assert grp.calls == [1, 1, 1]


def test_diverging_pool_plans_turn_the_round_off_everywhere():
    caps = (0, 40, 40)
    per = [(_kwargs(caps), 0), (_kwargs(caps), 40), (_kwargs(caps, n_req=1), 40)]
    out, grp = _decide(per, True)
    decs = [d for _p, d in out]
    assert decs[0] is not None and "diverged" in decs[0]
    assert len(set(decs)) == 1, "ranks disagree on the verdict"
    assert all(p is None for p, _d in out), "a rank would still retain"
    assert grp.calls == [1, 1, 1]


def test_diverging_caps_also_diverge():
    per = [(_kwargs((0, 40, 40)), 0), (_kwargs((0, 40, 30)), 40), (_kwargs((0, 40, 40)), 40)]
    out, _ = _decide(per, True)
    assert all("diverged" in d for _p, d in out if d) and all(d for _p, d in out)


def test_mutant_vote_without_the_digest_compare_lets_diverged_plans_run():
    mut = _mutant("flliper.srt.pdflip.l15_pool", "if len(fps) > 1 or (fps and nones):",
                  "if False:")
    per = [(_kwargs((0, 40, 40)), 0), (_kwargs((0, 40, 40)), 40),
           (_kwargs((0, 40, 40), n_req=1), 40)]
    grp = _Group(3)
    res = [None] * 3

    def work(r):
        kw, cap = per[r]
        planned = l15_retain.plan_round(
            candidates=kw["candidates"], slots_of=kw["slots_of"],
            anchor_slot_of=kw["anchor_slot_of"], caps_rows_by_rank=kw["caps_rows_by_rank"],
            cap_anchor_slots=8, prefix=PREFIX, epoch=8, log=lambda s: None, pool=True)
        res[r] = mut.agree_pool(None, planned.pool_fp, grp.gather_for(r))

    ts = [threading.Thread(target=work, args=(r,)) for r in range(3)]
    [t.start() for t in ts]
    [t.join(30) for t in ts]
    assert res == [None, None, None], "the mutant should wave diverged plans through"
    # the real vote refuses the very same input
    out, _ = _decide(per, True)
    assert all(d for _p, d in out)


def test_pool_off_vote_is_the_old_vote():
    caps = (0, 40, 40)
    out, grp = _decide([(_kwargs(caps), c) for c in caps], False)
    assert all(p is not None and p.pool_fp is None and p.guest_pieces is None
               for p, _d in out)
    # the old gather payload is the bare reason, not a pair
    sent = []
    A.decide_first(_kwargs(caps), False, lambda: 40, lambda: (1, PREFIX),
                   lambda v: sent.append(v) or [v], lambda s: None, lambda s: None,
                   pool=False)
    assert sent == [None]


# -- the park as the guest transport -------------------------------------------------


def _pool_env(tmp_path, on=True):
    env = {"FLLIPER_PDFLIP_L15_PARK_DIR": str(tmp_path)}
    if on:
        env["FLLIPER_PDFLIP_L15_POOL"] = "1"
    return env


def _manifest(rows=(12, 5, 6)):
    from flliper.srt.pdflip.l15_manifest import HoldSpan, Manifest

    sp = HoldSpan(rid="q0", depth=3, slots=(1, 2, 3), anchor_slot=1, l2_slots=(), l2_gens=())
    return Manifest(epoch=5, pid=1, spans=(sp,), rows_by_rank=tuple(rows), anchor_slots=1)


def test_park_on_follows_the_pool_switch():
    assert P.park_on({"FLLIPER_PDFLIP_L15_POOL": "1"}) is True
    assert P.park_on({"FLLIPER_PDFLIP_L15_PARK": "1"}) is True
    assert P.park_on({}) is False and P.park_on({"FLLIPER_PDFLIP_L15_POOL": "0"}) is False


def test_pooled_round_trip_checks_the_guest_rows_and_prints_the_lines(monkeypatch, tmp_path):
    m = _manifest()
    bufs = {0: T._bufs(0, fill=3), 1: T._bufs(1), 2: T._bufs(2)}
    orig0 = [b.clone() for b in bufs[0]]
    w, env, scheds = T._entries(monkeypatch, tmp_path, [m, m, m], bufs, [0, 15, 30])
    env.update(_pool_env(tmp_path))
    logs = []
    sent = T._run_ranks(lambda r: P.park_at_release(scheds[r], env, logs.append))
    assert sent[0] > 0 and sent[1] == 0 and sent[2] == 0
    outs = [x for x in logs if x.startswith("L15-POOL-OUT")]
    assert len(outs) == 3 and all("path=a2a" in x and "rounds=" in x and "GBps=" in x for x in outs)
    assert any("rank=0" in x and "bytes=%d" % sent[0] in x for x in outs)
    import json
    side = json.load(open(P.sidecar_path(0, env)))
    assert side["sums"], "the source checksum is in the sidecar"
    for b in bufs[0]:
        b.zero_()
    back = T._run_ranks(lambda r: P.park_back_at_wake(scheds[r], env, logs.append,
                                                      epoch=5, group_ok=True))
    assert back == [True, True, True]
    assert all(torch.equal(bufs[0][L][:12], orig0[L][:12]) for L in range(2))
    chk = [x for x in logs if x.startswith("L15-POOL-CHECK")]
    assert len(chk) == 3 and any("rank=0" in x and "bad=0" in x and "guest_bad=0" in x
                                 and "ok=0" not in x.split("rank=0")[1].split("bad")[0]
                                 for x in chk)
    assert sum(x.startswith("L15-POOL-BACK") and "reason=-" in x for x in logs) == 3


def test_a_guest_row_that_comes_back_wrong_is_a_group_fallback(monkeypatch, tmp_path):
    m = _manifest()
    bufs = {0: T._bufs(0, fill=3), 1: T._bufs(1), 2: T._bufs(2)}
    w, env, scheds = T._entries(monkeypatch, tmp_path, [m, m, m], bufs, [0, 15, 30])
    env.update(_pool_env(tmp_path))
    logs = []
    T._run_ranks(lambda r: P.park_at_release(scheds[r], env, logs.append))
    # the foreign segment is clobbered while TP0's pool is gone
    pieces, _ = P.park_plan([12, 5, 6], [0, 15, 30])
    for b in bufs[0]:
        b.zero_()
    for p in pieces:
        for b in bufs[p.dst]:
            b[p.dst_row:p.dst_row + p.rows] = 77
    back = T._run_ranks(lambda r: P.park_back_at_wake(scheds[r], env, logs.append,
                                                      epoch=5, group_ok=True))
    assert back == [False, False, False]          # EVERY rank falls back to L2
    assert any(x.startswith("L15-POOL-CHECK") and "bad=0" not in x for x in logs)
    assert any(x.startswith("L15-POOL-BACK") and "reason=failed" in x for x in logs)


def test_mutant_no_checksum_compare_hides_a_clobbered_guest(monkeypatch, tmp_path):
    mut = _mutant("flliper.srt.pdflip.l15_park", "if ck_bad:", "if False:")
    m = _manifest()
    bufs = {0: T._bufs(0, fill=3), 1: T._bufs(1), 2: T._bufs(2)}
    w, env, scheds = T._entries(monkeypatch, tmp_path, [m, m, m], bufs, [0, 15, 30])
    env.update(_pool_env(tmp_path))
    monkeypatch.setattr(mut, "_group_io", P._group_io)
    monkeypatch.setattr(mut, "_kv_buffers", P._kv_buffers)
    monkeypatch.setattr(mut, "_caps", P._caps)
    T._run_ranks(lambda r: P.park_at_release(scheds[r], env, lambda s: None))
    pieces, _ = P.park_plan([12, 5, 6], [0, 15, 30])
    for p in pieces:
        for b in bufs[p.dst]:
            b[p.dst_row:p.dst_row + p.rows] = 77
    for b in bufs[0]:
        b.zero_()
    back = T._run_ranks(lambda r: mut.park_back_at_wake(scheds[r], env, lambda s: None,
                                                        epoch=5, group_ok=True))
    assert back == [True, True, True], "the mutant should accept the clobbered rows"


def test_diverging_park_plans_start_no_collective(monkeypatch, tmp_path):
    m, m2 = _manifest((12, 5, 6)), _manifest((12, 6, 5))   # rank 1 planned another keep
    bufs = {r: T._bufs(r, fill=r + 1) for r in range(3)}
    w, env, scheds = T._entries(monkeypatch, tmp_path, [m, m2, m], bufs, [0, 15, 30])
    env.update(_pool_env(tmp_path))
    logs = []
    assert T._run_ranks(lambda r: P.park_at_release(scheds[r], env, logs.append)) == [None] * 3
    assert w.calls == 0
    assert any("park plan diverged" in x for x in logs)
    assert any(x.startswith("L15-POOL-OUT") and "path=none" in x for x in logs)
    assert not list(tmp_path.glob("pdflip_l15_park.*"))


def test_pool_off_park_is_unchanged_no_sums_no_pool_lines(monkeypatch, tmp_path):
    import json

    m = _manifest()
    bufs = {0: T._bufs(0, fill=3), 1: T._bufs(1), 2: T._bufs(2)}
    w, env, scheds = T._entries(monkeypatch, tmp_path, [m, m, m], bufs, [0, 15, 30])
    env["FLLIPER_PDFLIP_L15_PARK"] = "1"
    logs = []
    T._run_ranks(lambda r: P.park_at_release(scheds[r], env, logs.append))
    assert "sums" not in json.load(open(P.sidecar_path(0, env)))
    for b in bufs[0]:
        b.zero_()
    T._run_ranks(lambda r: P.park_back_at_wake(scheds[r], env, logs.append,
                                               epoch=5, group_ok=True))
    assert not [x for x in logs if x.startswith("L15-POOL")]


# -- the switch -----------------------------------------------------------------------


def test_switch_default_off_and_named_refusals():
    assert envs.FLLIPER_PDFLIP_L15_POOL.default is False
    on = {"FLLIPER_PDFLIP_L15_POOL": "1"}
    assert l15_pool.pool_on(on) and not l15_pool.pool_on({})
    assert "W-L15-DUAL" in l15_plan.refuse_dual(["--dual-layout"], on)
    assert "FLLIPER_PDFLIP_L15_POOL" in l15_plan.refuse_dual(["--dual-layout"], on)
    msg = l15_plan.refuse_not_27b("nf", on)
    assert "W-L15-27B-ONLY" in msg and "FLLIPER_PDFLIP_L15_POOL" in msg
    assert l15_plan.refuse_not_27b("qwen27b", on) is None
    assert "W-L15-POOL-MASTER" in l15_plan.refuse_pool_without_master(on)
    assert l15_plan.refuse_pool_without_master({**on, "FLLIPER_PDFLIP_L15": "1"}) is None
    assert l15_plan.refuse_pool_without_master({}) is None


def test_boot_line_modes():
    sh = l15_pool.boot_line([0, 7616, 1792], ["OVERRIDE"] * 3, 8)
    assert sh.endswith("mode=SHADOW(log-only, no behaviour change)")
    po = l15_pool.boot_line([0, 7616, 1792], ["OVERRIDE"] * 3, 8, mode=l15_pool.POOL_MODE)
    assert "mode=POOL(S2" in po and "total_mib=9408" in po


def test_scheduler_and_launcher_wiring():
    from flliper.srt.pdflip import launcher
    src = inspect.getsource(launcher)
    assert "refuse_pool_without_master(os.environ)" in src
    assert "mode=l15_pool.POOL_MODE" in src
    from flliper.srt.managers.scheduler_components import weight_updater as wu
    wsrc = inspect.getsource(wu)
    assert "_l15_pk.park_at_release(self.scheduler, os.environ, logger.info)" in wsrc
    assert "if _l15_pk.park_on(os.environ):" in wsrc


# -- plan and park cannot disagree; Q2 (the L2 duty stays in S2) ------------------------


def test_park_replans_exactly_the_guest_pieces_the_plan_checked():
    caps = (0, 16, 16)
    rp = _plan(caps, True, n_req=3)
    assert rp is not None
    kw = _kwargs(caps, n_req=3)
    man = l15_retain.manifest_of_plan(
        rp, candidates=kw["candidates"], l2_of=kw["l2_of"],
        anchor_l2_of=kw["anchor_l2_of"], l2_lanes_of=kw["l2_lanes_of"], epoch=8, pid=1)
    pieces, why = P.park_plan(list(man.rows_by_rank), list(caps))
    assert why is None and tuple(pieces) == rp.guest_pieces
    # the digest the park votes on is a function of the same lists
    assert l15_pool.plan_fingerprint(rp.hs.rids, man.rows_by_rank, caps, pieces) == rp.pool_fp


def test_q2_pool_keeps_the_l2_duty_of_the_cap0_rank():
    # the cap-0 rank still refuses a round whose owned held tokens have no L2
    # source (S5 would lift it; Q2 = S2-S4 keep the duty): the guests are the
    # transport, L2 is the fallback when the pool fails at the wake
    caps = (0, 40, 40)
    kw = _kwargs(caps)
    kw["l2_of"] = lambda rid: (tuple([-1] * 9 + [100, 101, 102]), (-1,) * 9 + (1, 1, 1))
    out, _ = _decide([(kw, 0), (dict(kw), 40), (dict(kw), 40)], True)
    decs = [d for _p, d in out]
    assert decs[0] is not None and "without an L2 source" in decs[0]
    assert len(set(decs)) == 1
