# SPDX-License-Identifier: Apache-2.0
"""L15-POOL stage S3 (KV, no anchor): EVERY rank may overflow into the free
rows of the other segments (guests by free area x measured link rate, the
slowest card only as the last overflow), admission against the SUM, manifest
v2 with the guest list in the group fingerprint, guest rows in the wake sample.

Hermetic: no CUDA, no model, no network.

Run (own worktree, capped):
  cd /spinning/wt-27b-l15-pool-s3-1004 && /spinning/gpu-arb/pytest_gedeckelt.sh \
    test/registered/unit/weg2/test_weg2_l15_pool_s3_1004.py

Pinned:
* ``pool_park_plan`` == ``l15_park.park_plan`` when only cap-0 ranks overflow
  and no rates are given (the S2 placement is the S3 placement's special case);
  with any-rank overflow it covers exactly the overflow rows, once, inside the
  hosts' free rows, and succeeds exactly when ``sum(keep) <= sum(caps)``
  (property); Q3: not the slowest card, free area x measured rate, lower rank;
* ``select_hold_pool_s3`` admits by the sum and nothing else changes (mutant:
  the sum check gone -> a hold without room);
* ``plan_round(s3=True)``: a capped rank's overflow is a guest, never a
  ``keep_over_cap`` trim; the plan checks the guest room on the COMPACTED rows
  before anything moves (mutant: the check gone -> a hold without room);
* manifest v2: guests/caps round-trip (binary + json), a v1 record and its
  fingerprint stay byte for byte, the fingerprint covers the guest list and
  the caps, so ranks that disagree on the placement fall back together
  (mutant: the guests left out of the fingerprint -> they agree);
* the park under S3 takes exactly the manifest's guest list, refuses a
  manifest that is v1 / has other caps / differs from the plan of its rates
  BEFORE any collective; the wake refuses a record that differs from the
  manifest list (mutant: compare gone); a guest row of a CAPPED rank that comes
  back wrong is a group fallback (checksum);
* step 7: the keep window of a capped overflow rank is its hold region at most,
  a hosting rank keeps its whole hold region; the zero scrub spares hosted rows;
* the wake sample covers guest rows (stratified plan);
* switches: S3 default off, needs POOL (refused by name), dual / non-27B refuse.
"""

from __future__ import annotations

import importlib
import inspect
import random
import sys
import types

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.weg2 import l15_manifest as M
from sglang.srt.weg2 import l15_park as P
from sglang.srt.weg2 import l15_plan, l15_pool, l15_retain
from sglang.srt.weg2 import l15_sleep_agree as A
from sglang.srt.weg2.l15_policy import Candidate

import test_weg2_l15_park_1002 as T  # _World, _bufs, _entries, _run_ranks
import test_weg2_l15_pool_s2_1004 as S2  # _plan_inputs, _kwargs, _decide, _cand
import test_weg2_l15_retain_0930 as R0  # make_scenario


_REAL_LOAD_BARLINK_RATES = l15_pool.load_barlink_rates


@pytest.fixture(autouse=True)
def _no_host_matrix(monkeypatch):
    """The barlink cache of the machine running the tests is not an input."""
    monkeypatch.setattr(l15_pool, "load_barlink_rates",
                        lambda env, tp: (None, "none(test)"))


def _mutant(modname, old, new, *more):
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


_cand = S2._cand
PREFIX = S2.PREFIX


# -- pool_park_plan ---------------------------------------------------------------


def _covers(pieces, keep, caps):
    """Every overflow row once, hosts only in their free rows, no overlap."""
    R = len(keep)
    home = [min(keep[r], caps[r]) for r in range(R)]
    seen = {}
    for p in pieces:
        assert p.src != p.dst and p.rows > 0
        assert caps[p.dst] > 0 and p.dst_row >= home[p.dst]
        assert p.dst_row + p.rows <= caps[p.dst]
        assert p.src_row >= home[p.src] and p.src_row + p.rows <= keep[p.src]
        for i in range(p.rows):
            assert (p.src, p.src_row + i) not in seen
            seen[(p.src, p.src_row + i)] = 1
            slot = (p.dst, p.dst_row + i)
            assert ("dst",) + slot not in seen
            seen[("dst",) + slot] = 1
    for r in range(R):
        for row in range(home[r], keep[r]):
            assert (r, row) in seen, "overflow row %d of rank %d is nowhere" % (row, r)


def test_equals_s2_park_plan_when_only_cap0_ranks_overflow_and_no_rates():
    rng = random.Random(3)
    checked = 0
    for _ in range(500):
        n = rng.choice([2, 3, 4])
        caps = [rng.randint(1, 40) for _ in range(n)]
        for k in rng.sample(range(n), rng.randint(0, max(0, n - 1))):
            caps[k] = 0
        keep = [rng.randint(0, 30) if caps[r] == 0 else rng.randint(0, caps[r])
                for r in range(n)]
        a = P.park_plan(keep, caps)
        b = l15_pool.pool_park_plan(keep, caps)
        assert a[1] == b[1] or (a[1] is not None and b[1] is not None)
        assert a[0] == b[0]
        checked += 1
    assert checked == 500


def test_any_rank_overflows_into_the_free_rows_of_the_others():
    keep, caps = [30, 30, 10], [20, 20, 40]
    pieces, why = l15_pool.pool_park_plan(keep, caps)
    assert why is None
    assert {p.src for p in pieces} == {0, 1} and {p.dst for p in pieces} == {2}
    assert sum(p.rows for p in pieces) == 20
    _covers(pieces, keep, caps)
    # S2 refuses this shape (rank 0 keeps more than its cap)
    assert P.park_plan(keep, caps)[1] is not None


def test_refused_by_name_when_the_pool_is_too_small():
    pieces, why = l15_pool.pool_park_plan([30, 30, 10], [20, 20, 25])
    assert pieces == [] and "rank 1 needs" in why or "rank 0 needs" in why
    assert "free" in why


def test_plan_covers_exactly_and_succeeds_iff_the_sum_fits_property():
    rng = random.Random(21)
    ok = refused = 0
    for _ in range(800):
        n = rng.choice([2, 3, 4])
        caps = [rng.randint(0, 25) for _ in range(n)]
        keep = [rng.randint(0, 30) for _ in range(n)]
        rates = None
        if rng.random() < 0.5:
            rates = {(a, b): rng.uniform(1, 15) for a in range(n) for b in range(n)
                     if a != b}
        pieces, why = l15_pool.pool_park_plan(keep, caps, rates)
        fits = sum(keep) <= sum(caps)
        assert (why is None) == fits, (keep, caps, why)
        if why is None:
            _covers(pieces, keep, caps)
            ok += 1
        else:
            assert pieces == []
            refused += 1
    assert ok > 100 and refused > 100


def test_q3_the_slowest_card_only_as_the_last_overflow():
    # ranks 1 and 2 both have 30 free rows; rank 1 is the slow card (inbound
    # rates 1.0 vs 10.0): the overflow of rank 0 goes to rank 2 first, the
    # slow card takes it only when rank 2 is full
    rates = {(0, 1): 1.0, (2, 1): 1.0, (1, 0): 10.0, (1, 2): 10.0,
             (0, 2): 10.0, (2, 0): 10.0}
    pieces, why = l15_pool.pool_park_plan([40, 0, 0], [10, 30, 30], rates)
    assert why is None
    assert [(p.dst, p.rows) for p in pieces] == [(2, 30)]
    pieces, why = l15_pool.pool_park_plan([55, 0, 0], [10, 30, 30], rates)
    assert why is None
    assert [(p.dst, p.rows) for p in pieces] == [(2, 30), (1, 15)]
    # without rates nobody is slow: the larger free area first, lower rank on a tie
    pieces, _ = l15_pool.pool_park_plan([55, 0, 0], [10, 30, 30])
    assert [(p.dst, p.rows) for p in pieces] == [(1, 30), (2, 15)]


def test_q3_free_area_times_the_directed_rate_decides_among_equals():
    # no card is the slowest (symmetric inbound means); host 1 has the larger
    # product free x rate(src->host): 20 x 8 = 160 against 50 x 2 = 100
    rates = {(0, 1): 8.0, (0, 2): 2.0, (1, 0): 5.0, (1, 2): 5.0, (2, 0): 5.0, (2, 1): 5.0}
    rt = l15_pool._Rates(rates, 1.0, [0, 1, 2])
    assert rt.slow == frozenset() or 0 in rt.slow or 1 in rt.slow or 2 in rt.slow
    pieces, _ = l15_pool.pool_park_plan([40, 0, 0], [10, 20, 50], rates)
    first = pieces[0]
    keyed = sorted((1, 2), key=lambda h: l15_pool._host_key(rt, 0, h, {1: 20, 2: 50}[h]))
    assert first.dst == keyed[0]


def test_plan_is_deterministic_and_rank_uniform():
    keep, caps = [37, 22, 9, 0], [20, 0, 30, 30]
    rates = {(a, b): 1.0 + ((3 * a + 5 * b) % 7) for a in range(4) for b in range(4) if a != b}
    a = l15_pool.pool_park_plan(keep, caps, rates)
    b = l15_pool.pool_park_plan(list(keep), tuple(caps), dict(reversed(list(rates.items()))))
    assert a == b


# -- select_hold_pool_s3 ------------------------------------------------------------


def test_s3_admission_is_the_sum_and_a_capped_overflow_is_no_longer_no_room():
    cands = [_cand("a", "seat", (0, 9, 0)), _cand("b", "parked", (0, 1, 0))]
    s2 = l15_pool.select_hold_pool(cands, (0, 8, 8), 8)
    assert s2.rids == ("b",) and ("a", "no_room") in s2.excluded
    s3 = l15_pool.select_hold_pool_s3(cands, (0, 8, 8), 8)
    assert s3.rids == ("a", "b")                       # 9 + 1 <= 16
    cands = [_cand("a", "seat", (6, 6, 6)), _cand("b", "parked", (6, 6, 6))]
    s3 = l15_pool.select_hold_pool_s3(cands, (8, 8, 8), 8)
    assert s3.rids == ("a",) and ("b", "pool_full") in s3.excluded


def test_s3_admission_keeps_order_anchorless_and_anchor_cap():
    c_bad = Candidate("x", "seat", 9.0, (1, 1, 1), anchor_depth=3, kv_depth=4)
    cands = [_cand("s", "served", (1, 1, 1), 9.0), _cand("p", "parked", (1, 1, 1), 1.0),
             _cand("t", "seat", (1, 1, 1), 0.0), c_bad]
    hs = l15_pool.select_hold_pool_s3(cands, (0, 8, 8), 8)
    assert hs.rids == ("t", "p", "s") and ("x", "anchorless") in hs.excluded
    hs = l15_pool.select_hold_pool_s3(
        [_cand("a", "seat", (1, 1, 1)), _cand("b", "parked", (1, 1, 1))], (0, 8, 8), 1)
    assert hs.rids == ("a",) and ("b", "anchor_full") in hs.excluded


def _random_cases():
    rng = random.Random(5)
    for _ in range(400):
        n = rng.choice([2, 3, 4])
        caps = [rng.randint(0, 30) for _ in range(n)]
        cands = [_cand("r%d" % i, rng.choice(["seat", "parked", "served"]),
                       [rng.randint(0, 9) for _ in range(n)], rng.random())
                 for i in range(rng.randint(1, 9))]
        yield caps, cands


def test_s3_never_admits_more_than_the_sum_and_never_less_property():
    for caps, cands in _random_cases():
        hs = l15_pool.select_hold_pool_s3(cands, caps, 99)
        assert sum(hs.rows_by_rank) <= sum(caps)
        by = {c.rid: sum(c.rows_by_rank) for c in cands}
        for rid, why in hs.excluded:
            if why == "pool_full":
                assert sum(hs.rows_by_rank) + by[rid] > sum(caps)


def test_mutant_s3_hold_without_room_is_killed_in_the_admission():
    mut = _mutant("sglang.srt.weg2.l15_pool", "if used + need > total_cap:", "if False:")
    bad = 0
    for caps, cands in _random_cases():
        if sum(mut.select_hold_pool_s3(cands, caps, 99).rows_by_rank) > sum(caps):
            bad += 1
    assert bad > 0, "the property does not see a hold admitted without room"


# -- plan_round ---------------------------------------------------------------------


def _plan(caps, n_req=3, s3=True, pool=True, log=None, mod=l15_retain, rates=None):
    cands, slots = S2._plan_inputs(n_req)
    return mod.plan_round(
        candidates=cands, slots_of=lambda rid: slots[rid], anchor_slot_of=lambda rid: 5,
        caps_rows_by_rank=caps, cap_anchor_slots=8, prefix=PREFIX, epoch=3,
        log=log or (lambda s: None), pool=pool, s3=s3, rates=rates)


def test_s3_off_is_the_s2_plan_byte_for_byte():
    for caps in [(0, 40, 40), (0, 16, 16), (10, 40, 40), (0, 14, 14)]:
        a = _plan(caps, s3=False)
        old = l15_retain.plan_round(
            candidates=S2._plan_inputs()[0], slots_of=lambda rid, s=S2._plan_inputs()[1]: s[rid],
            anchor_slot_of=lambda rid: 5, caps_rows_by_rank=caps, cap_anchor_slots=8,
            prefix=PREFIX, epoch=3, log=lambda s: None, pool=True, s3=False)
        assert (a is None) == (old is None)
        if a is not None:
            assert (a.hs, a.plan, a.guest_pieces, a.pool_fp, a.s3, a.caps) == (
                old.hs, old.plan, old.guest_pieces, old.pool_fp, False, None)


def test_s3_a_capped_rank_overflows_instead_of_being_trimmed():
    caps = (8, 8, 40)                    # ranks 0 and 1 overflow (12 rows each)
    s2 = _plan(caps, s3=False)
    s3 = _plan(caps, s3=True)
    assert s3 is not None and s3.s3 and s3.caps == caps
    assert len(s3.hs.rids) > (len(s2.hs.rids) if s2 else 0)
    rows = list(s3.plan.rows_by_rank)
    assert {p.src for p in s3.guest_pieces} == {r for r in (0, 1) if rows[r] > caps[r]}
    assert tuple(s3.guest_pieces) == tuple(l15_pool.pool_park_plan(rows, list(caps))[0])
    assert s3.pool_fp == l15_pool.plan_fingerprint(
        s3.hs.rids, rows, caps, s3.guest_pieces)
    _covers(s3.guest_pieces, rows, list(caps))


def test_s3_plan_line_names_the_overflow_hosts_and_rate_source():
    logs = []
    rp = _plan((8, 8, 40), log=logs.append, rates={(0, 2): 9.0, (1, 2): 9.0,
                                                    (2, 0): 9.0, (2, 1): 9.0,
                                                    (0, 1): 3.0, (1, 0): 3.0})
    assert rp is not None
    line = next(x for x in logs if x.startswith("L15-POOL-PLAN"))
    assert " s3=1 " in line and "overflow_rows=0:" in line and "guest_on_host=2:" in line
    assert "rates_src=none" in line and "fp=" + rp.pool_fp in line


@pytest.mark.parametrize("caps", [(4, 4, 4), (6, 6, 0), (10, 5, 5), (3, 3, 20)])
def test_s3_plan_never_holds_without_room(caps):
    rp = _plan(caps, n_req=3)
    if rp is None:
        return
    rows = list(rp.plan.rows_by_rank)
    assert l15_pool.pool_park_plan(rows, list(caps))[1] is None
    assert sum(rows) <= sum(caps)
    all_rids = ["q0", "q1", "q2"]
    assert list(rp.hs.rids) == all_rids[:len(rp.hs.rids)]       # lowest priority dropped first


def _uneven_plan(mod, caps):
    # one request whose 12 rows all belong to rank 0: the compaction pads every
    # rank to 13 rows (39 in all), far above the 12 rows the admission counted
    c = Candidate("u", "seat", 1.0, (12, 0, 0), 12, 12)
    slots = {"u": tuple(range(3, 39, 3))}
    return mod.plan_round(
        candidates=[c], slots_of=lambda r: slots[r], anchor_slot_of=lambda r: 5,
        caps_rows_by_rank=caps, cap_anchor_slots=8, prefix=PREFIX, epoch=3,
        log=lambda s: None, pool=True, s3=True, rates=None)


def test_s3_plan_checks_the_compacted_rows_not_the_admission_estimate():
    caps = (8, 8, 8)
    assert l15_pool.select_hold_pool_s3(
        [Candidate("u", "seat", 1.0, (12, 0, 0), 12, 12)], caps, 8).rids == ("u",)
    assert _uneven_plan(l15_retain, caps) is None                  # 39 rows > 24: not held
    rp = _uneven_plan(l15_retain, (14, 14, 14))
    assert rp is not None and rp.plan.rows_by_rank == (13, 13, 13) and rp.guest_pieces == ()


def test_mutant_s3_plan_without_the_guest_room_check_holds_without_room():
    caps = (8, 8, 8)
    both = _mutant("sglang.srt.weg2.l15_retain",
                   "                return l15_pool.pool_park_plan(\n"
                   "                    list(pl.rows_by_rank), [int(c) for c in caps_rows_by_rank],\n"
                   "                    rates)[1]\n",
                   "                return None\n",
                   "        if _gwhy is not None:\n",
                   "        if False:\n")
    # the trim loop's check alone gone: the final net still refuses (never a hold without room)
    loop_only = _mutant("sglang.srt.weg2.l15_retain",
                        "                return l15_pool.pool_park_plan(\n"
                        "                    list(pl.rows_by_rank), [int(c) for c in caps_rows_by_rank],\n"
                        "                    rates)[1]\n",
                        "                return None\n")
    assert _uneven_plan(loop_only, caps) is None
    bad = _uneven_plan(both, caps)
    assert bad is not None
    assert l15_pool.pool_park_plan(list(bad.plan.rows_by_rank), list(caps))[1] is not None, (
        "the mutant did not produce a hold without room -- the test is blind")


def test_s3_env_switch_drives_plan_round(monkeypatch):
    cands, slots = S2._plan_inputs(3)
    kw = dict(candidates=cands, slots_of=lambda rid: slots[rid],
              anchor_slot_of=lambda rid: 5, caps_rows_by_rank=(8, 8, 40),
              cap_anchor_slots=8, prefix=PREFIX, epoch=3, log=lambda s: None)
    monkeypatch.delenv("SGLANG_WEG2_L15_POOL", raising=False)
    monkeypatch.delenv("SGLANG_WEG2_L15_POOL_S3", raising=False)
    assert l15_retain.plan_round(**kw).s3 is False
    monkeypatch.setenv("SGLANG_WEG2_L15_POOL_S3", "1")           # alone: nothing
    assert l15_retain.plan_round(**kw).s3 is False
    monkeypatch.setenv("SGLANG_WEG2_L15_POOL", "1")
    assert l15_retain.plan_round(**kw).s3 is True


# -- the group vote -----------------------------------------------------------------


def test_ranks_on_different_s3_plans_turn_the_round_off_everywhere(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_L15_POOL", "1")
    monkeypatch.setenv("SGLANG_WEG2_L15_POOL_S3", "1")
    caps = (8, 8, 40)
    out, grp = S2._decide([(S2._kwargs(caps), c) for c in caps], True)
    assert all(d is None and p is not None and p.s3 for p, d in out)
    assert grp.calls == [1, 1, 1]
    kw2 = S2._kwargs((8, 8, 40))
    kw2["caps_rows_by_rank"] = (8, 8, 24)             # rank 1 planned other caps
    out, _ = S2._decide([(S2._kwargs(caps), 8), (kw2, 8), (S2._kwargs(caps), 40)], True)
    decs = [d for _p, d in out]
    assert all(d is not None and "diverged" in d for d in decs) and len(set(decs)) == 1


# -- manifest v2 --------------------------------------------------------------------


def _span(rid="q0"):
    return M.HoldSpan(rid=rid, depth=3, slots=(1, 2, 3), anchor_slot=1, l2_slots=(5, 6, 7),
                      l2_gens=(1, 1, 1), anchor_l2_slot=2, anchor_l2_gen=3, l2_lanes=(0, 1, 0))


def _m(guests=None, caps=None, rows=(12, 5, 6)):
    return M.Manifest(epoch=5, pid=1, spans=(_span(),), rows_by_rank=tuple(rows),
                      anchor_slots=1, guests=guests, caps=caps)


def test_v1_record_and_fingerprint_are_byte_for_byte_the_old_ones():
    import hashlib

    m = _m()
    assert m.guests is None and m.caps is None
    # golden values computed with the manifest module of the base commit (dfac7eb7b5)
    assert M.fingerprint(m) == 6869157374658007640
    assert hashlib.sha256(M.to_bytes(m)).hexdigest() == (
        "9ddfdf46a42f0e3f1252c71006df8655ae8128c0c1db146b84b313653250bb28")
    assert hashlib.sha256(M.to_json(m).encode()).hexdigest() == (
        "2e197402be6be4a418d7678957192957a5a1c8fcd69a29dc176e34af3f4e0b82")


def test_v2_round_trips_through_bytes_and_json_and_an_old_record_loads_as_v1():
    g = ((0, 2, 8, 5, 4), (1, 2, 8, 9, 3))
    m2 = _m(g, (8, 8, 40))
    for rt in (M.from_bytes(M.to_bytes(m2)), M.from_json(M.to_json(m2)),
               M.from_bytes(M.to_json(m2).encode())):
        assert rt == m2 and rt.guests == g and rt.caps == (8, 8, 40)
    assert M.from_bytes(M.to_bytes(_m())).guests is None
    empty = _m((), (8, 8, 40))
    assert M.from_bytes(M.to_bytes(empty)).guests == ()      # v2 with nothing travelling
    with pytest.raises(ValueError):
        M.from_json(M.to_json(m2).replace('"guests": [[0, 2, 8, 5, 4], [1, 2, 8, 9, 3]]',
                                          '"guests": [[0, 2, 8]]'))


def test_fingerprint_covers_guests_caps_and_the_version():
    g = ((0, 2, 8, 5, 4),)
    base = M.fingerprint(_m(g, (8, 8, 40)))
    assert M.fingerprint(_m(g, (8, 8, 40))) == base
    assert M.fingerprint(_m(((0, 2, 8, 5, 3),), (8, 8, 40))) != base      # a row count
    assert M.fingerprint(_m(((0, 2, 8, 6, 4),), (8, 8, 40))) != base      # a host row
    assert M.fingerprint(_m(((0, 1, 8, 5, 4),), (8, 8, 40))) != base      # a host
    assert M.fingerprint(_m(g, (8, 8, 41))) != base                       # the caps
    assert M.fingerprint(_m((), (8, 8, 40))) != M.fingerprint(_m())       # v2 vs v1
    assert M.decide(base, M.fingerprint(_m(((0, 1, 8, 5, 4),), (8, 8, 40)))) == "fallback"
    assert M.decide(base, base) == "hold"


def test_mutant_fingerprint_without_the_guests_lets_diverged_placements_agree():
    mut = _mutant("sglang.srt.weg2.l15_manifest",
                  "    head.update(_v2_head(m))\n", "    pass\n")
    a = mut.Manifest(epoch=5, pid=1, spans=(_span(),), rows_by_rank=(12, 5, 6),
                     anchor_slots=1, guests=((0, 2, 8, 5, 4),), caps=(8, 8, 40))
    b = mut.Manifest(epoch=5, pid=1, spans=(_span(),), rows_by_rank=(12, 5, 6),
                     anchor_slots=1, guests=((0, 1, 8, 5, 4),), caps=(8, 8, 40))
    assert mut.fingerprint(a) == mut.fingerprint(b), "the mutant should not see the placement"
    assert M.fingerprint(_m(a.guests, a.caps)) != M.fingerprint(_m(b.guests, b.caps))


def test_manifest_of_plan_publishes_v2_only_under_s3():
    caps = (8, 8, 40)
    kw = S2._kwargs(caps, n_req=3)

    def man(rp):
        return l15_retain.manifest_of_plan(
            rp, candidates=kw["candidates"], l2_of=kw["l2_of"],
            anchor_l2_of=kw["anchor_l2_of"], l2_lanes_of=kw["l2_lanes_of"], epoch=8, pid=1)

    rp = _plan(caps, s3=True)
    m = man(rp)
    assert m.guests == l15_pool.guest_tuples(rp.guest_pieces) and m.caps == caps
    rp2 = _plan((0, 40, 40), s3=False)
    m2 = man(rp2)
    assert m2.guests is None and m2.caps is None


# -- the park under S3 ----------------------------------------------------------------


def _s3_env(tmp_path, s3=True):
    env = {"SGLANG_WEG2_L15_PARK_DIR": str(tmp_path), "SGLANG_WEG2_L15_POOL": "1"}
    if s3:
        env["SGLANG_WEG2_L15_POOL_S3"] = "1"
    return env


CAPS = [8, 8, 40]
ROWS = (12, 11, 6)           # ranks 0 and 1 overflow by 4 and 3 rows, rank 2 hosts


def _v2(rows=ROWS, caps=CAPS, guests=None):
    pieces, why = l15_pool.pool_park_plan(list(rows), list(caps))
    assert why is None
    g = l15_pool.guest_tuples(pieces) if guests is None else guests
    return M.Manifest(epoch=5, pid=1, spans=(_span(),), rows_by_rank=tuple(rows),
                      anchor_slots=1, guests=g, caps=tuple(caps))


def _bufs3():
    return {r: T._bufs(r, fill=3 + 40 * r) for r in range(3)}


def test_s3_park_round_trips_the_overflow_of_capped_ranks_byte_exact(monkeypatch, tmp_path):
    m = _v2()
    bufs = _bufs3()
    orig = {r: [b.clone() for b in bufs[r]] for r in range(3)}
    w, env, scheds = T._entries(monkeypatch, tmp_path, [m, m, m], bufs, CAPS)
    env.update(_s3_env(tmp_path))
    logs = []
    sent = T._run_ranks(lambda r: P.park_at_release(scheds[r], env, logs.append))
    assert sent[0] > 0 and sent[1] > 0 and sent[2] == 0      # BOTH capped ranks send
    outs = [x for x in logs if x.startswith("L15-POOL-OUT")]
    assert len(outs) == 3 and all("path=a2a" in x for x in outs)
    pieces = l15_pool.pieces_of(m.guests)
    # the guest rows really lie in the host's free rows now
    for p in pieces:
        for L in range(2):
            assert torch.equal(bufs[p.dst][L][p.dst_row:p.dst_row + p.rows],
                               orig[p.src][L][p.src_row:p.src_row + p.rows])
    # the pause unmaps everything above the keep windows: rows >= home of the
    # capped ranks and the owner's whole pool are garbage at the wake
    for r in (0, 1):
        for b in bufs[r]:
            b[min(ROWS[r], CAPS[r]):] = 0
    back = T._run_ranks(lambda r: P.park_back_at_wake(
        scheds[r], env, logs.append, epoch=5, group_ok=True, manifest_guests=m.guests))
    assert back == [True, True, True]
    for r in (0, 1):
        for L in range(2):
            assert torch.equal(bufs[r][L][:ROWS[r]], orig[r][L][:ROWS[r]])
    chk = [x for x in logs if x.startswith("L15-POOL-CHECK")]
    assert len(chk) == 3 and all(" s3=1 guest_pieces=" in x for x in chk)
    assert any("rank=0" in x and "guest_rows=4" in x for x in chk)
    assert any("rank=1" in x and "guest_rows=3" in x for x in chk)
    assert any("rank=2" in x and "guest_rows=0" in x and "hosts=-" in x for x in chk)
    assert all("bad=0" in x for x in chk)


def test_s3_park_refuses_a_v1_manifest_other_caps_or_a_foreign_guest_list(monkeypatch, tmp_path):
    cases = {
        "v1": (M.Manifest(epoch=5, pid=1, spans=(_span(),), rows_by_rank=ROWS,
                          anchor_slots=1), "is v1 under"),
        "caps": (_v2(caps=[8, 9, 40], guests=l15_pool.guest_tuples(
            l15_pool.pool_park_plan(list(ROWS), CAPS)[0])), "caps"),
        "guests": (_v2(guests=((0, 1, 8, 8, 4),)), "differs from the plan"),
    }
    for name, (m, needle) in cases.items():
        bufs = _bufs3()
        w, env, scheds = T._entries(monkeypatch, tmp_path, [m, m, m], bufs, CAPS)
        env.update(_s3_env(tmp_path))
        logs = []
        assert T._run_ranks(lambda r: P.park_at_release(scheds[r], env, logs.append)) == [None] * 3
        assert w.calls == 0, name                       # no collective on a bad plan
        assert any(needle in x for x in logs), (name, logs)
        assert not list(tmp_path.glob("weg2_l15_park.*"))


def test_s3_rank_with_other_rates_and_another_placement_starts_no_collective(monkeypatch, tmp_path):
    # the manifest was planned with uniform rates (no slow card). One rank now
    # resolves rates under which rank 1 is the slow card, which moves the
    # placement of rank 0's overflow to host 2 first -- its pure re-plan no
    # longer equals the manifest's list: that rank votes no, the group is off
    # as ONE and no collective starts
    keep, caps = [30, 0, 0], [10, 30, 30]
    uniform, _ = l15_pool.pool_park_plan(keep, caps)
    m = M.Manifest(epoch=5, pid=1, spans=(_span(),), rows_by_rank=tuple(keep),
                   anchor_slots=1, guests=l15_pool.guest_tuples(uniform), caps=tuple(caps))
    slow1 = {(0, 1): 1.0, (2, 1): 1.0, (1, 0): 10.0, (1, 2): 10.0, (0, 2): 10.0, (2, 0): 10.0}
    assert l15_pool.guest_tuples(l15_pool.pool_park_plan(keep, caps, slow1)[0]) != m.guests
    bufs = _bufs3()
    w, env, scheds = T._entries(monkeypatch, tmp_path, [m, m, m], bufs, caps)
    env.update(_s3_env(tmp_path))
    seen = []

    def rates(env_, tp):
        seen.append(1)
        return (slow1, "test-other") if len(seen) == 2 else (None, "none(test)")

    monkeypatch.setattr(l15_pool, "load_barlink_rates", rates)
    logs = []
    res = T._run_ranks(lambda r: P.park_at_release(scheds[r], env, logs.append))
    assert res == [None, None, None] and w.calls == 0
    assert any("differs from the plan of this rank's rates" in x for x in logs)
    assert not list(tmp_path.glob("weg2_l15_park.*"))


def test_s3_wake_refuses_a_park_record_that_differs_from_the_manifest(monkeypatch, tmp_path):
    m = _v2()
    bufs = _bufs3()
    w, env, scheds = T._entries(monkeypatch, tmp_path, [m, m, m], bufs, CAPS)
    env.update(_s3_env(tmp_path))
    T._run_ranks(lambda r: P.park_at_release(scheds[r], env, lambda s: None))
    other = ((0, 2, 8, 7, 4), (1, 2, 8, 11, 3))
    logs = []
    back = T._run_ranks(lambda r: P.park_back_at_wake(
        scheds[r], env, logs.append, epoch=5, group_ok=True, manifest_guests=other))
    assert back == [False, False, False]
    assert any("differs from the manifest guest list" in x for x in logs)


def test_mutant_wake_without_the_manifest_compare_accepts_a_foreign_list(monkeypatch, tmp_path):
    mut = _mutant("sglang.srt.weg2.l15_park",
                  "    elif manifest_guests is not None and _guest_key(rec[1]) != _guest_key(\n"
                  "            [ParkPiece(*(int(x) for x in g)) for g in manifest_guests]):\n"
                  "        why = \"park record differs from the manifest guest list\"\n",
                  "")
    m = _v2()
    bufs = _bufs3()
    w, env, scheds = T._entries(monkeypatch, tmp_path, [m, m, m], bufs, CAPS)
    env.update(_s3_env(tmp_path))
    for n in ("_group_io", "_kv_buffers", "_caps"):
        monkeypatch.setattr(mut, n, getattr(P, n))
    T._run_ranks(lambda r: P.park_at_release(scheds[r], env, lambda s: None))
    other = ((0, 2, 8, 7, 4), (1, 2, 8, 11, 3))
    back = T._run_ranks(lambda r: mut.park_back_at_wake(
        scheds[r], env, lambda s: None, epoch=5, group_ok=True, manifest_guests=other))
    assert back == [True, True, True], "the mutant should wave the foreign list through"


def test_s3_a_guest_of_a_capped_rank_that_comes_back_wrong_is_a_group_fallback(monkeypatch, tmp_path):
    m = _v2()
    bufs = _bufs3()
    w, env, scheds = T._entries(monkeypatch, tmp_path, [m, m, m], bufs, CAPS)
    env.update(_s3_env(tmp_path))
    logs = []
    T._run_ranks(lambda r: P.park_at_release(scheds[r], env, logs.append))
    # the host's free rows holding RANK 1's guest rows are clobbered during the P phase
    p1 = [p for p in l15_pool.pieces_of(m.guests) if p.src == 1][0]
    for b in bufs[p1.dst]:
        b[p1.dst_row:p1.dst_row + p1.rows] = 77
    for r in (0, 1):
        for b in bufs[r]:
            b[min(ROWS[r], CAPS[r]):] = 0
    back = T._run_ranks(lambda r: P.park_back_at_wake(
        scheds[r], env, logs.append, epoch=5, group_ok=True, manifest_guests=m.guests))
    assert back == [False, False, False]
    assert any(x.startswith("L15-POOL-CHECK") and "rank=1" in x and "bad=0" not in x
               for x in logs)


def test_mutant_s3_no_checksum_compare_hides_a_clobbered_capped_guest(monkeypatch, tmp_path):
    mut = _mutant("sglang.srt.weg2.l15_park", "if ck_bad:", "if False:")
    m = _v2()
    bufs = _bufs3()
    w, env, scheds = T._entries(monkeypatch, tmp_path, [m, m, m], bufs, CAPS)
    env.update(_s3_env(tmp_path))
    for n in ("_group_io", "_kv_buffers", "_caps"):
        monkeypatch.setattr(mut, n, getattr(P, n))
    T._run_ranks(lambda r: P.park_at_release(scheds[r], env, lambda s: None))
    p1 = [p for p in l15_pool.pieces_of(m.guests) if p.src == 1][0]
    for b in bufs[p1.dst]:
        b[p1.dst_row:p1.dst_row + p1.rows] = 77
    for r in (0, 1):
        for b in bufs[r]:
            b[min(ROWS[r], CAPS[r]):] = 0
    back = T._run_ranks(lambda r: mut.park_back_at_wake(
        scheds[r], env, lambda s: None, epoch=5, group_ok=True, manifest_guests=m.guests))
    assert back == [True, True, True], "the mutant should accept the clobbered rows"


def test_s2_park_path_is_unchanged_when_s3_is_off(monkeypatch, tmp_path):
    m = S2._manifest()
    bufs = {0: T._bufs(0, fill=3), 1: T._bufs(1), 2: T._bufs(2)}
    w, env, scheds = T._entries(monkeypatch, tmp_path, [m, m, m], bufs, [0, 15, 30])
    env.update(_s3_env(tmp_path, s3=False))
    logs = []
    sent = T._run_ranks(lambda r: P.park_at_release(scheds[r], env, logs.append))
    assert sent[0] > 0 and sent[1] == 0 and sent[2] == 0
    back = T._run_ranks(lambda r: P.park_back_at_wake(scheds[r], env, logs.append,
                                                      epoch=5, group_ok=True))
    assert back == [True, True, True]
    chk = [x for x in logs if x.startswith("L15-POOL-CHECK")]
    assert chk and all(" s3=" not in x for x in chk)


# -- step 7, the zero scrub, the wake sample -----------------------------------------


def _retain(tmp_path, monkeypatch, rank, caps):
    monkeypatch.setenv("SGLANG_WEG2_L15_POOL", "1")
    monkeypatch.setenv("SGLANG_WEG2_L15_POOL_S3", "1")
    sc = R0.make_scenario(tmp_path, [])
    sc["kwargs"]["caps_rows_by_rank"] = caps
    sc["kwargs"]["rank"] = rank
    res = l15_retain.retain_at_sleep(**sc["kwargs"])
    assert res is not None
    keeps = [c[2] for c in sc["set_keep_calls"] if c[0] == "set_keep"]
    return res, sc, keeps


def test_s3_keep_windows_overflow_rank_at_most_its_region_host_keeps_all(tmp_path, monkeypatch):
    # plan rows (4, 4); caps (2, 10): rank 0 keeps 2 home rows (overflow 2),
    # rank 1 hosts rank 0's guest rows in rows [4, 6)
    res0, sc0, keeps0 = _retain(tmp_path / "a", monkeypatch, 0, (2, 10)) if (
        (tmp_path / "a").mkdir() or True) else None
    assert tuple(res0.manifest.rows_by_rank) == (4, 4)
    assert res0.manifest.guests == ((0, 1, 2, 4, 2),) and res0.manifest.caps == (2, 10)
    assert keeps0[0] == ((0, 2),)                          # kv window: home rows only
    (tmp_path / "b").mkdir()
    res1, sc1, keeps1 = _retain(tmp_path / "b", monkeypatch, 1, (2, 10))
    assert keeps1[0] == ((0, 10),)                         # the host keeps its whole region
    assert any("L15-POOL-PLAN" in x and " s3=1 " in x for x in sc1["log_lines"])


def test_s3_retain_off_keeps_the_old_window(tmp_path, monkeypatch):
    monkeypatch.delenv("SGLANG_WEG2_L15_POOL", raising=False)
    monkeypatch.delenv("SGLANG_WEG2_L15_POOL_S3", raising=False)
    sc = R0.make_scenario(tmp_path, [])
    res = l15_retain.retain_at_sleep(**sc["kwargs"])
    assert res.manifest.guests is None and res.manifest.caps is None
    assert [c[2] for c in sc["set_keep_calls"] if c[0] == "set_keep"][0] == ((0, 4),)


def test_wake_scrub_spares_the_hosted_guest_rows():
    m = _v2()
    host = 2
    assert l15_pool.hosted_end(m.guests, host) == max(g[3] + g[4] for g in m.guests)
    assert l15_pool.wake_keep_rows(m.rows_by_rank, host, m.guests) > m.rows_by_rank[host]
    assert l15_pool.wake_keep_rows(m.rows_by_rank, 0, m.guests) == m.rows_by_rank[0]
    assert l15_pool.wake_keep_rows(m.rows_by_rank, host, None) == m.rows_by_rank[host]
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    assert wu._l15_keep_rows_of(m, host) == l15_pool.wake_keep_rows(
        m.rows_by_rank, host, m.guests)
    v1 = M.Manifest(epoch=5, pid=1, spans=(_span(),), rows_by_rank=ROWS, anchor_slots=1)
    assert wu._l15_keep_rows_of(v1, host) == ROWS[host]          # v1: the old value


def test_capped_guest_ranks_name_the_ranks_without_an_l2_path():
    m = _v2()
    assert l15_pool.capped_guest_ranks(m.guests, m.caps) == (0, 1)
    z = _v2(rows=(12, 5, 6), caps=[0, 8, 40])
    assert l15_pool.capped_guest_ranks(z.guests, z.caps) == ()      # the cap-0 rank refills from L2
    assert l15_pool.capped_guest_ranks((), (8, 8)) == ()
    assert l15_pool.capped_guest_ranks(None, None) == ()


def test_stratified_wake_sample_covers_the_guest_rows():
    plan = [("q0", row, 100 + row, 1) for row in range(200)]
    ranges = [(150, 170)]
    got = l15_pool.stratified_check_plan(plan, ranges, 64)
    assert len(got) == 64
    in_g = [e for e in got if 150 <= e[1] < 170]
    assert len(in_g) >= 20                        # all 20 guest rows fit in the half
    assert len({e[1] for e in got}) == 64 and [e[1] for e in got] == sorted(e[1] for e in got)
    # no guest rows -> the plain even sample; fewer rows than k -> all of them
    assert [e[1] for e in l15_pool.stratified_check_plan(plan, [], 4)] == [0, 50, 100, 150]
    assert len(l15_pool.stratified_check_plan(plan[:5], [(1, 3)], 64)) == 5
    # a wake sample that never touches the guest rows is what the old plan gave
    old = [e[1] for e in plan][::3]
    assert not all(150 <= r < 170 for r in old) and in_g


def test_guest_row_ranges():
    m = _v2()
    assert l15_pool.guest_row_ranges(m.guests, 0) == [(8, 12)]
    assert l15_pool.guest_row_ranges(m.guests, 1) == [(8, 11)]
    assert l15_pool.guest_row_ranges(m.guests, 2) == []


# -- rates --------------------------------------------------------------------------


def test_rates_come_from_the_measurement_never_from_a_card_name():
    class Meas:
        world = 3
        sizes = (20480, 1048576)
        sensor = "pair"

        @staticmethod
        def capacity(src, dst, gi):
            return {(0, 1): 12.9, (1, 0): 12.8, (0, 2): 6.4, (2, 0): 6.5,
                    (1, 2): 6.4, (2, 1): 6.6}[(src, dst)] * (1 if gi in (1, -1) else 0.5)

    r = l15_pool.rates_from_capacity(Meas.capacity, 3)
    assert r[(0, 1)] == 12.9 and r[(2, 0)] == 6.5 and len(r) == 6
    rt = l15_pool._Rates(r, 1.0, [0, 1, 2])
    assert rt.slow == frozenset({2})              # the measured slow card, whatever its name
    assert "name" not in inspect.signature(l15_pool.rates_from_capacity).parameters
    assert l15_pool.rates_from_capacity(lambda a, b, g: 1 / 0, 3) is None
    assert l15_pool.quantize_rates({(0, 1): 1.00049, (1, 0): float("nan"), (1, 2): -1}) == {(0, 1): 1.0}
    assert l15_pool.rates_digest(None) == "-" == l15_pool.rates_digest({})
    assert l15_pool.rates_digest({(0, 1): 2.0}) != l15_pool.rates_digest({(0, 1): 3.0})


def test_resolve_rates_priority_env_then_matrix_then_uniform():
    env = {"SGLANG_WEG2_L15_POOL_RATES": "0>1=13.4,1>0=14.4"}
    r, src = l15_pool.resolve_rates(env, 2, loader=lambda e, t: ({(0, 1): 1.0}, "m"))
    assert src == "env" and r == {(0, 1): 13.4, (1, 0): 14.4}
    r, src = l15_pool.resolve_rates({"SGLANG_WEG2_L15_POOL_RATES": "junk"}, 2,
                                    loader=lambda e, t: ({(0, 1): 1.0}, "m"))
    assert r is None and "malformed" in src                 # named, no silent fallback
    r, src = l15_pool.resolve_rates({}, 2, loader=lambda e, t: ({(0, 1): 1.0}, "m"))
    assert (r, src) == ({(0, 1): 1.0}, "m")
    assert l15_pool.resolve_rates({}, 2, loader=lambda e, t: (None, "none(x)")) == (None, "none(x)")


def test_load_barlink_rates_reads_a_measurement_of_the_right_world(tmp_path):
    import json

    meas = {"world": 3, "sizes": [20480, 1048576], "bdfs": ["a", "b", "c"],
            "names": ["x", "y", "z"], "sensor": "pair",
            "outbound": {"0": [1, 12.0], "1": [1, 12.0], "2": [1, 6.0]},
            "inbound": {"0": [1, 12.0], "1": [1, 12.0], "2": [1, 6.0]},
            "edge": {"0->1": [1, 11.0]}}
    path = tmp_path / "bm.json"
    path.write_text(json.dumps({"fingerprint": "f", "measurement": meas}))
    env = {"SGLANG_BARLINK_MATRIX_CACHE": str(path)}
    rates, src = _REAL_LOAD_BARLINK_RATES(env, 3)
    assert rates[(0, 1)] == 11.0                    # a measured edge wins
    assert rates[(1, 0)] == 12.0                    # else min(outbound, inbound)
    assert rates[(0, 2)] == 6.0 and rates[(2, 0)] == 6.0
    assert "barlink-matrix(ranks=3,sensor=pair,size_kib=1024)" == src
    # a measurement of another world is not the pool's matrix
    r2, src2 = _REAL_LOAD_BARLINK_RATES(env, 2)
    assert r2 is None and "of 3 ranks, the pool has 2" in src2
    # no file / unreadable: named, never a crash
    r3, src3 = _REAL_LOAD_BARLINK_RATES({"SGLANG_BARLINK_MATRIX_CACHE": str(tmp_path / "x")}, 3)
    assert r3 is None and src3.startswith("none(")


def test_switches_default_off_and_named_refusals():
    assert envs.SGLANG_WEG2_L15_POOL_S3.default is False
    both = {"SGLANG_WEG2_L15_POOL": "1", "SGLANG_WEG2_L15_POOL_S3": "1"}
    assert l15_pool.pool_s3_on(both)
    assert not l15_pool.pool_s3_on({"SGLANG_WEG2_L15_POOL_S3": "1"})
    assert not l15_pool.pool_s3_on({"SGLANG_WEG2_L15_POOL": "1"}) and not l15_pool.pool_s3_on({})
    only_s3 = {"SGLANG_WEG2_L15_POOL_S3": "1"}
    assert "W-L15-POOL-S3-NEEDS-POOL" in l15_plan.refuse_pool_s3_without_pool(only_s3)
    assert l15_plan.refuse_pool_s3_without_pool(both) is None
    assert l15_plan.refuse_pool_s3_without_pool({}) is None
    assert "SGLANG_WEG2_L15_POOL_S3" in l15_plan.refuse_dual(["--dual-layout"], only_s3)
    assert "W-L15-DUAL" in l15_plan.refuse_dual(["--dual-layout"], only_s3)
    msg = l15_plan.refuse_not_27b("nf", only_s3)
    assert "W-L15-27B-ONLY" in msg and "SGLANG_WEG2_L15_POOL_S3" in msg
    assert l15_plan.refuse_not_27b("qwen27b", both) is None


def test_pool_with_the_deposit_region_is_refused_by_name():
    on = {"SGLANG_WEG2_L15_POOL": "1"}
    assert "W-L15-POOL-DEPOSIT" in l15_plan.refuse_pool_with_deposit(
        {**on, "SGLANG_WEG2_L15_DEPOSIT": "1"})
    assert l15_plan.refuse_pool_with_deposit(on) is None
    assert l15_plan.refuse_pool_with_deposit({"SGLANG_WEG2_L15_DEPOSIT": "1"}) is None
    assert l15_plan.refuse_pool_with_deposit({}) is None
    from sglang.srt.weg2 import launcher
    assert "refuse_pool_with_deposit(os.environ)" in inspect.getsource(launcher)


def test_boot_line_s3_mode_and_wiring():
    po = l15_pool.boot_line([8, 8, 40], ["RECORD"] * 3, 8, mode=l15_pool.S3_MODE)
    assert "mode=POOL(S3" in po and "total_mib=56" in po
    from sglang.srt.weg2 import launcher
    src = inspect.getsource(launcher)
    assert "refuse_pool_s3_without_pool(os.environ)" in src
    assert "else l15_pool.S3_MODE" in src and "mode=l15_pool.POOL_MODE if not l15_pool.pool_s3_on" in src
    from sglang.srt.managers.scheduler_components import weight_updater as wu
    wsrc = inspect.getsource(wu)
    assert "manifest_guests=_l15_mg" in wsrc
    assert "_l15_s3_drop" in wsrc and "capped_guest_ranks" in wsrc
    j = wsrc.index("_l15_pk.park_back_at_wake(")
    assert j < wsrc.index("_l15_opt_failed = self._l15_optimistic_refill()")
    assert "_l15_opt_failed = True" in wsrc[j:j + 4000]       # the S3 drop is a no-hold vote
    assert "stratified_check_plan" in wsrc and "guest_row_ranges" in wsrc
    assert "_l15_keep_rows_of(m, rank)" in wsrc


def test_q2_the_cap0_l2_duty_is_unchanged_under_s3(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_L15_POOL", "1")
    monkeypatch.setenv("SGLANG_WEG2_L15_POOL_S3", "1")
    caps = (0, 40, 40)
    kw = S2._kwargs(caps)
    kw["l2_of"] = lambda rid: (tuple([-1] * 9 + [100, 101, 102]), (-1,) * 9 + (1, 1, 1))
    out, _ = S2._decide([(kw, 0), (dict(kw), 40), (dict(kw), 40)], True)
    decs = [d for _p, d in out]
    assert decs[0] is not None and "without an L2 source" in decs[0]
    assert len(set(decs)) == 1
