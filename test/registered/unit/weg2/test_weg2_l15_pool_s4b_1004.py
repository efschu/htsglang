# SPDX-License-Identifier: Apache-2.0
"""L15-POOL stage S4b: the DYNAMIC anchor count (user idea 04.10.2026 ~17:50Z:
"falls noch kv frei waere, koennte ein anker diesen platz im vram verdraengen;
dann schrumpft zwar max kv, aber wenn der eh leer waere, so what?").

Hermetic: no CUDA, no model, no network.

Run (own worktree, capped):
  cd /spinning/wt-27b-l15-pool-s4b-1004 && /spinning/gpu-arb/pytest_gedeckelt.sh \
    test/registered/unit/weg2/test_weg2_l15_pool_s4b_1004.py

Pinned:
* the anchors of a rank WITH a home segment beyond ``anchor_cap`` are byte pieces in
  the KV hold rows -- the own home segment first, then foreign segments (Q3: free
  area x rate, the slowest card last, never a card name); the ranks without a home
  segment hold all their anchors as pieces (S4, unchanged);
* admission plans KV rows AND all anchor bytes of ALL ranks against ONE row sum, in
  the candidate order: no anchor count cap, the anchor consumes rows, KV gets the
  rest, the total is never exceeded (mutants: anchors of the capped ranks not counted
  -> a hold without room; the sum check removed; the cursor not advanced -> KV and
  anchor on the same rows);
* the plan places KV guests and anchors on the SAME free rows and cursors, never a row
  twice (property test over random pools), whole anchors per piece, all or nothing;
* S4b off (anchor_cap None) = the S4 plan / digest / manifest record byte for byte
  (golden values computed with the S4 commit 18e0fa407f);
* the digest and the manifest carry the anchor cap (mutant: digest without it lets
  ranks on another cap agree); ``L15-POOL-PLAN`` names s4b=1, the overflow pieces and
  rows and ``kv_rows_left``;
* park: owner==host pieces are a local copy (no collective), foreign pieces travel,
  both come back byte exact at the wake with the source checksum (mutant: the compare
  dropped -> a clobbered overflow share goes unnoticed); the wake scrub spares the
  rows that host overflow anchors, also in the OWNER'S OWN segment; a capped rank that
  owns overflow anchors drops the hold when the park-back did not land;
* switches: S4b default off, needs POOL + S3 + S4 (refused by name), dual / non-27B
  refuse it.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import random
import sys
import types

import pytest
import torch

from sglang.srt.weg2 import l15_manifest as M
from sglang.srt.weg2 import l15_park as P
from sglang.srt.weg2 import l15_plan, l15_pool, l15_pool_anchor as PA, l15_retain

import test_weg2_l15_park_1002 as T  # _World, _bufs, _entries, _run_ranks
import test_weg2_l15_pool_s2_1004 as S2  # _plan_inputs, PREFIX
import test_weg2_l15_pool_s4_1004 as S4  # _cand, _mviews, _wire, _span, _s4_env ...

PREFIX = S2.PREFIX


@pytest.fixture(autouse=True)
def _no_host_matrix(monkeypatch):
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


# -- 1. admission: one row sum over KV and ALL anchor bytes --------------------------


def test_anchor_rows_of_index_charges_a_capped_rank_only_beyond_the_cap():
    caps, ab, rb = [0, 20, 20], (24, 12, 12), 8     # shares: 3 / 2 / 2 rows
    # below the cap only the rank without a home segment pays (S4's a_rows)
    assert PA.anchor_rows_of_index(0, caps, ab, rb, 2) == 3
    assert PA.anchor_rows_of_index(1, caps, ab, rb, 2) == 3
    # from anchor `cap` on the capped ranks' shares are pieces too: 3 + 2 + 2
    assert PA.anchor_rows_of_index(2, caps, ab, rb, 2) == 7
    assert PA.anchor_rows_of_index(0, caps, ab, rb, 0) == 7
    assert PA.anchor_rows_of_index(0, caps, ab, rb, 2) == \
        PA.anchor_guest_rows_per_anchor(caps, ab, rb)


def test_s4b_has_no_anchor_count_cap_the_rows_decide():
    # 6 requests of 6 KV rows, three ranks with a home segment of 40 rows (120),
    # anchor_cap 2, shares 24/12/12 B at 8 B rows (3+2+2 = 7 rows per overflow anchor)
    cands = [S4._cand("r%d" % i, (2, 2, 2)) for i in range(6)]
    caps, ab = [40, 40, 40], (24, 12, 12)
    s4 = PA.select_hold_pool_s4(cands, caps, 2, ab, 8)
    s4b = PA.select_hold_pool_s4b(cands, caps, ab, 8, 2)
    assert len(s4.rids) == 2, "S4: the count cap"
    assert s4b.rids == tuple("r%d" % i for i in range(6)), "S4b: all fit the rows"
    # the budget: 6x6 KV + anchors 0,1 free + 4 x 7 overflow rows <= 120
    pool, kv, a, left = PA.pool_row_budget(caps, [12, 12, 12], [(0, 0, 2, 4, 12, 12, 96)])
    assert (pool, kv) == (120, 36) and left == 120 - 36 - 12


def test_s4b_the_anchor_displaces_kv_rows_and_the_total_is_never_exceeded():
    # 3 ranks x 10 rows = 30 rows; every request 6 KV rows; overflow anchor 7 rows
    # from anchor 1 on (cap 1): r0 6, r1 6+7, r2 6+7, r3 6+7 would be 58 > 30
    cands = [S4._cand("r%d" % i, (2, 2, 2)) for i in range(5)]
    caps, ab = [10, 10, 10], (24, 12, 12)
    hs = PA.select_hold_pool_s4b(cands, caps, ab, 8, 1)
    used = 0
    for i, rid in enumerate(hs.rids):
        used += 6 + PA.anchor_rows_of_index(i, caps, ab, 8, 1)
    assert used <= sum(caps)
    assert hs.rids == ("r0", "r1")                      # 6 + (6+7) = 19; r2: 32 > 30
    ex = dict(hs.excluded)
    assert ex["r2"] == "anchor_full" and ex["r3"] == "anchor_full"   # the KV alone fits
    # KV-only budget would have held 5 (30 rows): the anchors took the rest
    assert len(l15_pool.select_hold_pool_s3(cands, caps, 8).rids) == 5


def test_s4b_without_a_capped_rank_is_s4_in_the_admission():
    cands = [S4._cand("a", (4, 6, 6)), S4._cand("b", (4, 6, 6)), S4._cand("c", (4, 6, 6)),
             S4._cand("d", (1, 1, 0), "parked")]
    caps, ab = [0, 20, 20], (24, 12, 12)
    # anchor_cap larger than the candidate count: no capped rank ever overflows
    s4b = PA.select_hold_pool_s4b(cands, caps, ab, 8, 99)
    s4 = PA.select_hold_pool_s4(cands, caps, 99, ab, 8)
    assert s4b == s4


def test_mutant_capped_overflow_not_counted_admits_a_hold_without_room():
    cands = [S4._cand("a", (4, 4, 4)), S4._cand("b", (4, 4, 4))]
    caps, ab = [14, 14, 14], (240, 120, 120)     # 30 + 15 + 15 rows per overflow anchor
    real = PA.select_hold_pool_s4b(cands, caps, ab, 8, 0)
    assert real.rids == ()                        # 12 KV + 60 anchor rows > 42
    mut = _mutant("sglang.srt.weg2.l15_pool_anchor",
                  "               if r < len(caps) and int(b) > 0 and (int(caps[r]) <= 0 or over))",
                  "               if r < len(caps) and int(b) > 0 and int(caps[r]) <= 0)")
    held = mut.select_hold_pool_s4b(cands, caps, ab, 8, 0)
    assert held.rids, "the mutant must admit a request whose overflow anchor has no room"
    _p, _g, why, code = PA.pool_park_plan_s4([4, 4, 4], caps, ab, 8, 1, anchor_cap=0)
    assert why and code == "anchor_full"          # the exact plan is the second guard


def test_mutant_sum_check_removed_exceeds_the_pool():
    cands = [S4._cand("a", (4, 4, 4)), S4._cand("b", (4, 4, 4)), S4._cand("c", (4, 4, 4))]
    caps, ab = [11, 11, 11], (24, 12, 12)         # 33 rows; a = 12 + 7, b = 12 more + 7
    real = PA.select_hold_pool_s4b(cands, caps, ab, 8, 0)
    assert real.rids == ("a",)                    # b: 31 fits the KV, 38 > 33 with its anchor
    mut = _mutant("sglang.srt.weg2.l15_pool_anchor",
                  "        if need + a_rows + used > total_cap:",
                  "        if False:")
    held = mut.select_hold_pool_s4b(cands, caps, ab, 8, 0)
    assert len(held.rids) == 2 > len(real.rids)
    keep = [4 * len(held.rids)] * 3               # 24 KV rows + 12 packed anchor rows = 36 > 33
    _p, _g, why, code = PA.pool_park_plan_s4(keep, caps, ab, 8, len(held.rids), anchor_cap=0)
    assert why and code in ("pool_full", "anchor_full"), "the exact plan refuses the overshoot"
    assert real.rids and sum(
        4 * 3 + PA.anchor_rows_of_index(i, caps, ab, 8, 0) for i in range(len(real.rids))
    ) <= sum(caps)


# -- 2. the plan: KV guests, then the anchors on the SAME free rows -----------------


def _check_plan4b(keep, caps, ab, rb, n, cap, pieces, guests):
    """KV guests + anchor pieces: inside the hosts' free rows, no row twice, each
    owner's overflow anchors [lo, n) covered exactly once (lo = 0 without a home
    segment, anchor_cap with one), owner==host only for an owner with a home
    segment, nbytes consistent, home first."""
    R = len(keep)
    home = [min(keep[r], caps[r]) for r in range(R)]
    used = {}
    for p in pieces:
        for i in range(p.rows):
            k = (p.dst, p.dst_row + i)
            assert k not in used, "KV rows overlap %r" % (k,)
            used[k] = "kv"
    cover = {}
    for (o, h, a_lo, n_a, h_lo, h_rows, nb) in guests:
        assert n_a > 0 and caps[h] > 0
        assert nb == n_a * ab[o] and h_rows == -(-nb // rb)
        if o == h:
            assert caps[o] > 0, "a rank without a home segment has no own segment"
        assert h_lo >= home[h] and h_lo + h_rows <= caps[h]
        for i in range(h_rows):
            k = (h, h_lo + i)
            assert k not in used, "anchor rows overlap %r (%s)" % (k, used[k])
            used[k] = "anchor"
        for a in range(a_lo, a_lo + n_a):
            assert (o, a) not in cover
            cover[(o, a)] = h
    for o in range(R):
        if ab[o] <= 0 or n == 0:
            continue
        lo = 0 if caps[o] <= 0 else min(cap, n)
        assert sorted(a for (oo, a) in cover if oo == o) == list(range(lo, n)), (o, cover)
    # HOME FIRST: an owner with a home segment spills to a foreign host only when its
    # own free rows (after the KV guests and its own pieces) could not take one more
    # whole anchor -- pass 1 only ever touches the owner's own segment
    for o in range(R):
        if caps[o] > 0 and any(g[0] == o and g[1] != o for g in guests):
            rest = (caps[o] - home[o] - sum(p.rows for p in pieces if p.dst == o)
                    - sum(g[5] for g in guests if g[0] == o and g[1] == o))
            assert (rest * rb) // ab[o] < 1, ("spilled with room at home", o, rest)


def test_s4b_capped_owner_overflow_goes_home_first_then_foreign():
    # all three ranks have a home segment (20 rows), anchor_cap 2, n = 5 anchors
    caps, ab, rb = [20, 20, 20], (24, 12, 12), 8
    # rank 0 keeps 6 rows, 14 free: its anchors 2..4 (3 x 3 rows) lie at HOME
    keep = [6, 6, 5]
    pieces, guests, why, code = PA.pool_park_plan_s4(keep, caps, ab, rb, 5, anchor_cap=2)
    assert why is None and code is None and pieces == []
    _check_plan4b(keep, caps, ab, rb, 5, 2, pieces, guests)
    own = [g for g in guests if g[0] == 0]
    assert own == [(0, 0, 2, 3, 6, 9, 72)], "home first: own free rows, after the home rows"
    assert all(g[0] == g[1] for g in guests), "every owner fits at home"
    # rank 0 keeps 18 rows: only 2 free rows, not even one 3-row anchor -> foreign
    keep = [18, 6, 5]
    pieces, guests, why, _c = PA.pool_park_plan_s4(keep, caps, ab, rb, 5, anchor_cap=2)
    assert why is None
    _check_plan4b(keep, caps, ab, rb, 5, 2, pieces, guests)
    assert all(g[1] != 0 for g in guests if g[0] == 0), "no room at home: a guest"
    assert {g[0] for g in guests if g[0] == g[1]} == {1, 2}
    # a partly fitting home segment: 8 free rows = 2 whole anchors (6 rows), the 3rd foreign
    keep = [12, 6, 5]
    pieces, guests, why, _c = PA.pool_park_plan_s4(keep, caps, ab, rb, 5, anchor_cap=2)
    assert why is None
    _check_plan4b(keep, caps, ab, rb, 5, 2, pieces, guests)
    g0 = [g for g in guests if g[0] == 0]
    assert g0[0] == (0, 0, 2, 2, 12, 6, 48) and g0[1][1] != 0 and g0[1][2:4] == (4, 1)


def test_s4b_below_the_cap_nothing_overflows_and_the_ranks_without_home_hold_all():
    caps, ab, rb = [0, 20, 20], (24, 12, 12), 8
    keep = [6, 6, 5]
    # n <= cap: no capped rank overflows -> exactly the S4 anchors (rank 0 only)
    p4, g4, w4, _ = PA.pool_park_plan_s4(keep, caps, ab, rb, 2)
    pb, gb, wb, _ = PA.pool_park_plan_s4(keep, caps, ab, rb, 2, anchor_cap=2)
    assert w4 is None and wb is None
    assert (l15_pool.guest_tuples(p4), g4) == (l15_pool.guest_tuples(pb), gb)
    # n > cap: rank 0 (no home) still holds ALL n, the capped ones the rest
    pb, gb, wb, _ = PA.pool_park_plan_s4(keep, caps, ab, rb, 4, anchor_cap=2)
    assert wb is None
    _check_plan4b(keep, caps, ab, rb, 4, 2, pb, gb)
    assert sorted(a for g in gb if g[0] == 0 for a in range(g[2], g[2] + g[3])) == [0, 1, 2, 3]
    assert sorted(a for g in gb if g[0] == 1 for a in range(g[2], g[2] + g[3])) == [2, 3]


def test_s4b_off_plan_is_the_s4_plan_byte_for_byte():
    # golden from the S4 commit 18e0fa407f (pool_park_plan_s4([9,6,5],[0,20,20],..))
    pieces, guests, why, code = PA.pool_park_plan_s4(
        [9, 6, 5], [0, 20, 20], (24, 12, 12), 8, 3)
    assert why is None and code is None
    assert [(p.src, p.dst, p.src_row, p.dst_row, p.rows) for p in pieces] == [(0, 2, 0, 5, 9)]
    assert guests == [(0, 1, 0, 3, 6, 9, 72)]


def test_s4b_plan_refuses_by_name_and_all_or_nothing():
    caps, ab, rb = [4, 4, 4], (24, 12, 12), 8
    keep = [3, 3, 3]
    # overflow anchors need rows nobody has: named anchor_full, nothing returned
    pieces, guests, why, code = PA.pool_park_plan_s4(keep, caps, ab, rb, 6, anchor_cap=0)
    assert why and code == "anchor_full" and pieces == [] and guests == []
    assert "has no room" in why
    # length mismatches
    assert PA.pool_park_plan_s4([1, 1], caps, ab, rb, 1, anchor_cap=0)[2]
    # no anchors: nothing
    assert PA.pool_park_plan_s4(keep, caps, ab, rb, 0, anchor_cap=0)[1] == []


def test_s4b_property_valid_cover_no_double_occupation_or_named_refusal():
    rnd = random.Random(1004)
    placed = refused = local = foreign = 0
    for _ in range(600):
        R = rnd.choice([2, 3, 4])
        caps = [rnd.choice([0, 0, 6, 12, 20, 33]) for _ in range(R)]
        if not any(c > 0 for c in caps):
            caps[0] = 12
        rb = rnd.choice([4, 8, 16])
        ab = tuple(rnd.choice([0, 12, 24, 40, 100]) for _ in range(R))
        keep = [rnd.randint(0, (c + 8) if c else 10) for c in caps]
        n = rnd.randint(0, 7)
        cap = rnd.randint(0, 4)
        rates = ({(a, b): rnd.choice([1.0, 2.0, 0.5]) for a in range(R) for b in range(R)
                  if a != b} if rnd.random() < 0.5 else None)
        pieces, guests, why, code = PA.pool_park_plan_s4(keep, caps, ab, rb, n, rates,
                                                         anchor_cap=cap)
        if why is not None:
            refused += 1
            assert code in ("pool_full", "anchor_full") and pieces == [] and guests == []
            continue
        placed += 1
        _check_plan4b(keep, caps, ab, rb, n, cap, pieces, guests)
        local += sum(1 for g in guests if g[0] == g[1])
        foreign += sum(1 for g in guests if g[0] != g[1])
        pool, kv, a, left = PA.pool_row_budget(caps, keep, guests)
        assert left >= 0, (caps, keep, guests)       # KV + anchors never exceed the sum
        assert a == sum(g[5] for g in guests)
        again = PA.pool_park_plan_s4(keep, caps, ab, rb, n, rates, anchor_cap=cap)
        assert (again[0], again[1]) == (pieces, guests), "deterministic / rank-uniform"
    assert placed > 100 and refused > 50 and local > 10 and foreign > 10


def test_s4b_the_plan_never_uses_more_rows_than_the_segments_have():
    # the pool as ONE budget: kv rows held (compacted keep rows, home or guest) plus
    # every anchor piece row never exceed the sum of the segments
    rnd = random.Random(7)
    for _ in range(400):
        R = 3
        caps = [rnd.choice([0, 10, 25]) for _ in range(R)]
        if not any(caps):
            continue
        rb, ab = 8, (rnd.choice([24, 40]), rnd.choice([12, 24]), rnd.choice([12, 24]))
        keep = [rnd.randint(0, 30) for _ in range(R)]
        n, cap = rnd.randint(0, 6), rnd.randint(0, 3)
        pieces, guests, why, _c = PA.pool_park_plan_s4(keep, caps, ab, rb, n, anchor_cap=cap)
        if why is not None:
            continue
        pool, kv, a, left = PA.pool_row_budget(caps, keep, guests)
        assert left >= 0, (caps, keep, guests)


def test_mutant_cursor_not_advanced_puts_kv_and_anchor_on_the_same_rows():
    mut = _mutant("sglang.srt.weg2.l15_pool_anchor",
                  "                cursor[owner] += rows\n                free[owner] -= rows\n"
                  "                a += n",
                  "                free[owner] -= rows\n                a += n")
    caps, ab, rb = [20, 20, 20], (24, 12, 12), 8
    keep = [18, 6, 5]        # rank 0 spills into hosts that already hold their own pieces
    real = PA.pool_park_plan_s4(keep, caps, ab, rb, 6, anchor_cap=2)
    pieces, guests, why, _c = mut.pool_park_plan_s4(keep, caps, ab, rb, 6, anchor_cap=2)
    assert why is None and real[2] is None
    _check_plan4b(keep, caps, ab, rb, 6, 2, real[0], real[1])        # the real plan: valid
    # the mutant hands the SAME start row to two pieces of one host: the check is red
    with pytest.raises(AssertionError, match="overlap"):
        _check_plan4b(keep, caps, ab, rb, 6, 2, pieces, guests)


def test_mutant_anchor_starting_inside_the_home_rows_is_caught_by_the_same_check():
    # a plan that ignores the owner's home rows (cursor from 0) overlaps its own KV
    mut = _mutant("sglang.srt.weg2.l15_pool_anchor",
                  "    # pass 1: HOME FIRST -- a rank with a home segment puts as many whole overflow\n",
                  "    cursor = {r: 0 for r in cursor}\n"
                  "    # pass 1: HOME FIRST -- a rank with a home segment puts as many whole overflow\n")
    caps, ab, rb = [20, 20, 20], (24, 12, 12), 8
    keep = [6, 6, 5]
    pieces, guests, why, _c = mut.pool_park_plan_s4(keep, caps, ab, rb, 5, anchor_cap=2)
    assert why is None
    with pytest.raises(AssertionError):
        _check_plan4b(keep, caps, ab, rb, 5, 2, pieces, guests)


def test_q3_the_slowest_card_hosts_overflow_anchors_only_as_the_last_overflow():
    caps, ab, rb = [20, 20, 20], (24, 12, 12), 8
    keep = [20, 6, 6]                      # rank 0 has no free rows -> foreign
    # rank 1 is the slowest card (lowest inbound rate): the anchors prefer rank 2
    rates = {(0, 1): 1.0, (1, 0): 1.0, (2, 1): 1.0, (1, 2): 1.0, (0, 2): 5.0, (2, 0): 5.0}
    _p, g, why, _c = PA.pool_park_plan_s4(keep, caps, ab, rb, 4, rates, anchor_cap=1)
    assert why is None
    assert [x[1] for x in g if x[0] == 0] == [2], "not the slowest card, free x rate"
    # keyed by cap and rank, never by a card name
    src = inspect.getsource(PA._anchor_guests_s4b)
    assert "3080" not in src and "5090" not in src and "card" not in src.lower()


# -- 3. plan_round: digest, manifest, marker --------------------------------------------

CTX = PA.AnchorCtx((24, 12, 12), 8, "test", anchor_cap=2)
CTX_S4 = PA.AnchorCtx((24, 12, 12), 8, "test")


def _plan(caps, n_req=4, ctx=CTX, log=None):
    cands, slots = S2._plan_inputs(n_req)
    return l15_retain.plan_round(
        candidates=cands, slots_of=lambda rid: slots[rid],
        anchor_slot_of=lambda rid: 10 + int(str(rid)[-1]),
        caps_rows_by_rank=caps, cap_anchor_slots=2, prefix=PREFIX, epoch=3,
        log=log or (lambda s: None), pool=True, s3=True, rates=None, s4=True, anchor_ctx=ctx)


def test_plan_round_s4b_holds_more_anchors_than_the_cap_and_names_the_overflow():
    logs = []
    rp = _plan((24, 24, 24), n_req=4, log=logs.append)
    assert rp is not None and rp.anchor_ctx is CTX
    n = rp.a_h - 1
    assert n == 4 and n > CTX.anchor_cap, "more anchors than anchor_cap are held"
    rows = list(rp.plan.rows_by_rank)
    kv, ag, why, _c = PA.pool_park_plan_s4(rows, [24, 24, 24], CTX.bytes_by_rank, 8, n,
                                           anchor_cap=2)
    assert why is None and tuple(ag) == rp.anchor_guests and tuple(kv) == rp.guest_pieces
    _check_plan4b(rows, [24, 24, 24], CTX.bytes_by_rank, 8, n, 2, kv, ag)
    assert rp.pool_fp == PA.plan_fingerprint_s4(
        rp.hs.rids, rows, (24, 24, 24), kv, ag, CTX.bytes_by_rank, 8, n, anchor_cap=2)
    line = [x for x in logs if x.startswith("L15-POOL-PLAN")][0]
    assert " s3=1 " in line and " s4=1 " in line and " s4b=1 " in line
    n_p, n_r, n_b, n_h = PA.anchor_overflow_totals(ag, [24, 24, 24])
    assert "anchor_overflow_pieces=%d" % n_p in line and n_p > 0
    assert "anchor_overflow_rows=%d" % n_r in line
    pool, kvr, a, left = PA.pool_row_budget([24, 24, 24], rows, ag)
    assert "kv_rows_left=%d" % left in line and left >= 0
    assert "pool_rows=72" in line and "anchor_cap=2" in line


def test_plan_round_s4b_pool_row_budget_is_dynamic_the_anchor_shrinks_max_kv():
    # same pool, anchor_cap 99 (no overflow at all) vs anchor_cap 0 (every anchor
    # takes rows): the rows left for KV shrink by exactly the anchor rows
    logs_a, logs_b = [], []
    a = _plan((24, 24, 24), n_req=3, log=logs_a.append,
              ctx=PA.AnchorCtx((24, 12, 12), 8, "t", anchor_cap=99))
    b = _plan((24, 24, 24), n_req=3, log=logs_b.append,
              ctx=PA.AnchorCtx((24, 12, 12), 8, "t", anchor_cap=0))
    la = int([x for x in logs_a if x.startswith("L15-POOL-PLAN")][0]
             .split("kv_rows_left=")[1].split()[0])
    lb = int([x for x in logs_b if x.startswith("L15-POOL-PLAN")][0]
             .split("kv_rows_left=")[1].split()[0])
    assert a.a_h == b.a_h and lb < la
    n_rows_b = sum(g[5] for g in b.anchor_guests)
    n_rows_a = sum(g[5] for g in a.anchor_guests)
    assert la - lb == n_rows_b - n_rows_a


def test_plan_round_s4_ctx_without_a_cap_is_the_s4_round_byte_for_byte():
    s4 = _plan((0, 40, 40), n_req=3, ctx=CTX_S4)
    assert s4 is not None and s4.anchor_ctx.anchor_cap is None
    n = s4.a_h - 1
    rows = list(s4.plan.rows_by_rank)
    kv, ag, why, _c = PA.pool_park_plan_s4(rows, [0, 40, 40], CTX_S4.bytes_by_rank, 8, n)
    assert why is None and tuple(ag) == s4.anchor_guests and tuple(kv) == s4.guest_pieces
    assert s4.pool_fp == PA.plan_fingerprint_s4(
        s4.hs.rids, rows, (0, 40, 40), kv, ag, CTX_S4.bytes_by_rank, 8, n)
    logs = []
    _plan((0, 40, 40), n_req=3, ctx=CTX_S4, log=logs.append)
    assert not any(" s4b=1" in x for x in logs)


def test_digest_s4_golden_and_the_cap_is_part_of_the_s4b_digest():
    pieces = [l15_pool.ParkPiece(0, 1, 6, 9, 6)]
    g = [(0, 1, 0, 3, 15, 9, 72), (0, 2, 0, 1, 5, 3, 24)]
    args = (["a", "b"], [9, 6, 5], [0, 20, 20], pieces, g, (24, 12, 12), 8, 3)
    # golden computed with plan_fingerprint_s4 of the S4 commit 18e0fa407f
    assert PA.plan_fingerprint_s4(*args) == "f48da10c77b7c8b6"
    a2 = PA.plan_fingerprint_s4(*args, anchor_cap=2)
    a3 = PA.plan_fingerprint_s4(*args, anchor_cap=3)
    assert len({PA.plan_fingerprint_s4(*args), a2, a3}) == 3


def test_ranks_on_another_anchor_cap_turn_the_round_off_everywhere():
    a = _plan((24, 24, 24), ctx=PA.AnchorCtx((24, 12, 12), 8, "x", anchor_cap=2))
    b = _plan((24, 24, 24), ctx=PA.AnchorCtx((24, 12, 12), 8, "x", anchor_cap=1))
    assert a.pool_fp != b.pool_fp
    assert l15_pool.agree_pool(None, a.pool_fp, lambda v: [v, (None, b.pool_fp)])
    assert l15_pool.agree_pool(None, a.pool_fp, lambda v: [v, v]) is None


def test_mutant_digest_without_the_cap_lets_ranks_on_another_cap_agree():
    mut = _mutant("sglang.srt.weg2.l15_pool_anchor",
                  "        parts = parts + ((\"s4b\", int(anchor_cap)),)",
                  "        parts = parts")
    args = (["a"], [4, 4, 4], [20, 20, 20], [], [(0, 0, 2, 3, 4, 9, 72)], (24, 12, 12), 8, 5)
    assert mut.plan_fingerprint_s4(*args, anchor_cap=2) == mut.plan_fingerprint_s4(
        *args, anchor_cap=1), "the mutant cannot tell two caps apart"
    assert PA.plan_fingerprint_s4(*args, anchor_cap=2) != PA.plan_fingerprint_s4(
        *args, anchor_cap=1)


def test_s4b_ctx_resolved_from_the_switch_only():
    env = {"SGLANG_WEG2_L15_POOL": "1", "SGLANG_WEG2_L15_POOL_S3": "1",
           PA.POOL_S4_ENV: "1", "SGLANG_WEG2_L15_ANCHOR_CAP": "3"}
    ctx = PA.AnchorCtx((24, 12, 12), 8, "env")
    src = inspect.getsource(PA.resolve_anchor_ctx)
    assert "pool_s4b_on(env)" in src and "dataclasses.replace" in src
    assert PA.pool_s4b_on(env) is False
    assert PA.pool_s4b_on({**env, PA.POOL_S4B_ENV: "1"}) is True
    assert ctx.anchor_cap is None


# -- 4. manifest v2 + S4b ----------------------------------------------------------------


def _m4b(cap=2, ag=((0, 0, 2, 3, 6, 9, 72),), ab=(24, 12, 12), rb=8,
         guests=((0, 1, 2, 4, 2),), caps=(2, 10, 10)):
    return M.Manifest(epoch=5, pid=1, spans=(S4._span(),), rows_by_rank=(12, 5, 6),
                      anchor_slots=6, guests=guests, caps=caps, anchor_guests=ag,
                      anchor_bytes=ab, anchor_row_bytes=rb, anchor_cap=cap)


def test_s4_record_and_fingerprint_are_byte_for_byte_the_old_ones():
    # golden computed with the manifest module of the S4 commit 18e0fa407f
    m = M.Manifest(epoch=5, pid=1, spans=(S4._span(),), rows_by_rank=(12, 5, 6),
                   anchor_slots=3, guests=((0, 1, 2, 4, 2),), caps=(2, 10, 10),
                   anchor_guests=((0, 1, 0, 2, 4, 3, 24),), anchor_bytes=(24, 12, 12),
                   anchor_row_bytes=8)
    assert m.anchor_cap is None
    assert M.fingerprint(m) == -1331254946122221380
    assert hashlib.sha256(M.to_bytes(m)).hexdigest() == (
        "f36ce509dcc68e4d1e81865f5bf60d24300b94bb4fb6f499f0b0afc3c02f2607")
    assert hashlib.sha256(M.to_json(m).encode()).hexdigest() == (
        "e101e546481e49e62d9d4eb61a736d8697be30bcff377ebb18fde3f24885a0fe")


def test_s4b_round_trips_and_the_cap_is_in_the_group_fingerprint():
    m = _m4b()
    for back in (M.from_bytes(M.to_bytes(m)), M.from_json(M.to_json(m))):
        assert back.anchor_cap == 2 and back.anchor_guests == m.anchor_guests
        assert back == m
    assert M.fingerprint(_m4b(cap=2)) != M.fingerprint(_m4b(cap=3))
    s4 = _m4b(cap=None)
    assert M.from_bytes(M.to_bytes(s4)).anchor_cap is None
    assert M.fingerprint(s4) != M.fingerprint(m)


def test_mutant_manifest_fingerprint_without_the_cap():
    mut = _mutant("sglang.srt.weg2.l15_manifest",
                  "        if m.anchor_cap is not None:\n            out[\"anchor_cap\"] = int(m.anchor_cap)\n",
                  "")
    assert mut.fingerprint(_m4b(cap=2)) == mut.fingerprint(_m4b(cap=3))
    assert M.fingerprint(_m4b(cap=2)) != M.fingerprint(_m4b(cap=3))


def test_manifest_of_plan_publishes_the_cap_only_under_s4b():
    cands, _slots = S2._plan_inputs(4)

    def man(rp):
        return l15_retain.manifest_of_plan(
            rp, candidates=cands, l2_of=lambda rid: ((1, 2), (1, 1)),
            anchor_l2_of=lambda rid: (1, 1), l2_lanes_of=lambda rid: (0, 0),
            epoch=3, pid=1)

    m4b = man(_plan((24, 24, 24)))
    assert m4b.anchor_cap == 2 and m4b.anchor_guests
    m4 = man(_plan((0, 40, 40), n_req=3, ctx=CTX_S4))
    assert m4.anchor_cap is None and m4.anchor_guests is not None


# -- 5. the park: local copy for the own segment, collective for foreign pieces ---------

CAPS = [20, 20, 20]
KEEP = (18, 6, 5)      # rank 0: 2 free rows (foreign), ranks 1/2: own segment
AB = S4.AB
RB = S4.RB
N_A = 5
CAP = 2


def _man(keep=KEEP, caps=CAPS, ab=AB, rb=RB, n=N_A, cap=CAP):
    kv, ag, why, _ = PA.pool_park_plan_s4(list(keep), list(caps), ab, rb, n, anchor_cap=cap)
    assert why is None, why
    return M.Manifest(epoch=5, pid=1, spans=(S4._span(),), rows_by_rank=tuple(keep),
                      anchor_slots=n + 1, guests=l15_pool.guest_tuples(kv),
                      caps=tuple(caps), anchor_guests=PA.anchor_guest_tuples(ag),
                      anchor_bytes=tuple(ab), anchor_row_bytes=rb, anchor_cap=cap)


def _env(tmp_path, s4b=True):
    env = S4._s4_env(tmp_path)
    if s4b:
        env[PA.POOL_S4B_ENV] = "1"
    return env


def _wipe_overflow(mv, cap=CAP):
    """What the release leaves: the mamba slots above the hold region are gone
    (the hold region [0, cap+1) survives)."""
    for r in range(3):
        for v in mv[r]:
            v[cap + 1:].fill_(99)


def test_s4b_park_round_trips_local_and_foreign_overflow_anchors_byte_exact(
        monkeypatch, tmp_path):
    m = _man()
    local = [g for g in m.anchor_guests if g[0] == g[1]]
    foreign = [g for g in m.anchor_guests if g[0] != g[1]]
    assert local and foreign, "the fixture needs an own-segment AND a guest piece"
    assert {g[0] for g in local} == {1, 2} and {g[0] for g in foreign} == {0}
    mv = S4._mviews()
    orig = {r: [v.clone() for v in mv[r]] for r in range(3)}
    bufs = {r: T._bufs(r, fill=3 + 40 * r) for r in range(3)}
    korig = {r: [b.clone() for b in bufs[r]] for r in range(3)}
    w, env, scheds = T._entries(monkeypatch, tmp_path, [m, m, m], bufs, CAPS)
    env.update(_env(tmp_path))
    S4._wire(monkeypatch, mv)
    logs = []
    sent = T._run_ranks(lambda r: P.park_at_release(scheds[r], env, logs.append))
    assert all(x is not None for x in sent), logs
    # collectives only for the foreign pieces (one per piece and 16 MiB block)
    step = P.chunk_rows(RB, env)
    want_calls = sum(-(-g[5] // step) for g in foreign)
    assert w.calls == want_calls, (w.calls, want_calls)
    # the anchor bytes lie in the host rows, the home rows of every host untouched
    for g in m.anchor_guests:
        exp = S4._expected_rows(orig, g)
        for L in range(2):
            got = bufs[g[1]][L][g[4]:g[4] + g[5]]
            assert torch.equal(got, exp[:, 4 * L:4 * L + 4]), g
    for r in range(3):
        for L in range(2):
            lo = min(KEEP[r], CAPS[r])
            assert torch.equal(bufs[r][L][:lo], korig[r][L][:lo]), (r, L)
    _wipe_overflow(mv)
    back = T._run_ranks(lambda r: P.park_back_at_wake(
        scheds[r], env, logs.append, epoch=5, group_ok=True,
        manifest_guests=m.guests, manifest_anchor_guests=m.anchor_guests))
    assert back == [True, True, True], logs
    for r in range(3):
        for v, o in zip(mv[r], orig[r]):
            assert torch.equal(v[1:N_A + 1], o[1:N_A + 1]), r      # every anchor is back
    chk = [x for x in logs if x.startswith("L15-POOL-CHECK")]
    assert len(chk) == 3 and all("anchor_bad=0" in x for x in chk)
    assert any("rank=1" in x and "anchor_pieces=%d" % sum(1 for g in m.anchor_guests
                                                           if g[0] == 1) in x for x in chk)
    assert not list(tmp_path.glob("weg2_l15_park.*"))


def _clobber(monkeypatch, tmp_path, which, park_mod=P):
    m = _man()
    mv = S4._mviews()
    bufs = {r: T._bufs(r, fill=3 + 40 * r) for r in range(3)}
    w, env, scheds = T._entries(monkeypatch, tmp_path, [m, m, m], bufs, CAPS)
    env.update(_env(tmp_path))
    S4._wire(monkeypatch, mv)
    if park_mod is not P:
        for n in ("_group_io", "_kv_buffers", "_caps"):
            monkeypatch.setattr(park_mod, n, getattr(P, n))
    T._run_ranks(lambda r: P.park_at_release(scheds[r], env, lambda s: None))
    g = [x for x in m.anchor_guests if (x[0] == x[1]) == (which == "local")][0]
    for L in range(2):
        bufs[g[1]][L][g[4]] = 77             # a row of the piece clobbered at the host
    _wipe_overflow(mv)
    logs = []
    back = T._run_ranks(lambda r: park_mod.park_back_at_wake(
        scheds[r], env, logs.append, epoch=5, group_ok=True,
        manifest_guests=m.guests, manifest_anchor_guests=m.anchor_guests))
    return back, logs


@pytest.mark.parametrize("which", ["local", "foreign"])
def test_s4b_an_overflow_share_that_comes_back_wrong_is_a_group_fallback(
        monkeypatch, tmp_path, which):
    back, logs = _clobber(monkeypatch, tmp_path, which)
    assert back == [False, False, False]
    assert any("anchor round-trip checksum" in x for x in logs)
    assert any("anchor_bad=" in x and "anchor_bad=0" not in x for x in logs)


def test_mutant_wake_without_the_anchor_compare_hides_a_clobbered_overflow_share(
        monkeypatch, tmp_path):
    mut = _mutant("sglang.srt.weg2.l15_park",
                  "            if a_bad and err is None:", "            if False:")
    back, _logs = _clobber(monkeypatch, tmp_path, "local", park_mod=mut)
    assert back == [True, True, True], "the mutant lets the wrong bytes through"
    (tmp_path / "real").mkdir()
    back_real, _ = _clobber(monkeypatch, tmp_path / "real", "local")
    assert back_real == [False, False, False]


def test_s4b_park_refuses_before_any_collective_when_switch_and_manifest_disagree(
        monkeypatch, tmp_path):
    for name, m, s4b in (("switch on, manifest S4", _man(cap=None, n=3), True),
                         ("switch off, manifest S4b", _man(), False)):
        mv = S4._mviews()
        bufs = {r: T._bufs(r, fill=3 + 40 * r) for r in range(3)}
        w, env, scheds = T._entries(monkeypatch, tmp_path, [m, m, m], bufs, CAPS)
        env.update(_env(tmp_path, s4b=s4b))
        S4._wire(monkeypatch, mv)
        logs = []
        res = T._run_ranks(lambda r: P.park_at_release(scheds[r], env, logs.append))
        assert res == [None] * 3, (name, logs)
        assert w.calls == 0 and any("S4b switch" in x for x in logs), (name, logs)


def test_s4b_a_manifest_anchor_cap_changes_what_the_plan_is_checked_against(
        monkeypatch, tmp_path):
    # a manifest that claims another cap than the pieces were planned with is not
    # run: the plan of the manifest's own cap differs from the listed pieces
    good = _man()
    bad = M.Manifest(**{**good.__dict__, "anchor_cap": 3})
    mv = S4._mviews()
    bufs = {r: T._bufs(r, fill=3 + 40 * r) for r in range(3)}
    w, env, scheds = T._entries(monkeypatch, tmp_path, [bad, bad, bad], bufs, CAPS)
    env.update(_env(tmp_path))
    S4._wire(monkeypatch, mv)
    logs = []
    res = T._run_ranks(lambda r: P.park_at_release(scheds[r], env, logs.append))
    assert res == [None] * 3 and w.calls == 0
    assert any("anchor guest list differs" in x for x in logs)


def test_s4b_local_pieces_post_no_collective_foreign_ones_do():
    src = inspect.getsource(PA.run_anchor_park)
    assert "if owner == host:" in src and "continue" in src.split("if owner == host:")[1][:1200]


# -- 6. wake: the scrub, the keep window, the drop ---------------------------------------


def test_wake_scrub_spares_the_rows_that_host_overflow_anchors_in_the_own_segment():
    rows = (6, 6, 5)
    guests = ()                                       # no KV guests at all
    ag = ((0, 0, 2, 3, 6, 9, 72),)                    # rank 0's overflow in its OWN rows 6..15
    assert l15_pool.wake_keep_rows(rows, 0, (), ag) == 15
    assert l15_pool.wake_keep_rows(rows, 0, (), None) == 6          # the S3 value
    assert l15_pool.wake_keep_rows(rows, 1, (), ag) == 6
    assert PA.hosted_end_anchor(ag, 0) == 15
    mut = _mutant("sglang.srt.weg2.l15_pool",
                  "    if anchor_guests:\n        end = max(end,", "    if False:\n        end = max(end,")
    assert mut.wake_keep_rows(rows, 0, (), ag) == 6, "the mutant scrubs the own-segment anchors"
    from sglang.srt.managers.scheduler_components import weight_updater as WU

    assert WU._l15_keep_rows_of(_m4b(ag=((0, 0, 2, 3, 6, 9, 72),)), 0) == 15


def test_retain_keeps_the_whole_hold_region_of_an_own_segment_host_and_clips_the_mamba_window(
        tmp_path, monkeypatch):
    # rows (4,4), caps (16,16), anchor_cap 0: every anchor of rank 0 and 1 is an
    # overflow piece in the owner's own segment; the mamba keep window stops at the
    # hold region (cap + 1 = 1 slot), the KV window is the whole hold region
    monkeypatch.setenv("SGLANG_WEG2_L15_POOL_S4B", "1")
    ctx = PA.AnchorCtx((24, 12), 8, "test", anchor_cap=0)
    res, sc, keeps = S4._retain4(tmp_path / "a", monkeypatch, 0, (16, 16), ctx)
    m = res.manifest
    assert m.anchor_cap == 0 and m.anchor_guests
    assert all(g[0] == g[1] for g in m.anchor_guests), "everything fits at home"
    assert m.anchor_slots - 1 > ctx.anchor_cap
    kv_win = [k for k in keeps if k and k[0][1] == 16]
    assert kv_win, keeps
    mamba_wins = [k for k in keeps if k and k[0][1] < 16]
    assert mamba_wins and all(k == ((0, 1),) for k in mamba_wins), keeps
    line = [x for x in sc["log_lines"] if x.startswith("L15-POOL-PLAN")][0]
    assert " s4b=1 " in line and "anchor_overflow_pieces=" in line and "kv_rows_left=" in line


def test_a_capped_rank_that_owns_overflow_anchors_drops_the_hold_when_park_back_failed():
    ag = ((0, 0, 2, 3, 6, 9, 72), (2, 1, 0, 1, 3, 2, 12))
    assert PA.capped_anchor_overflow_ranks(ag, (20, 20, 0)) == (0,)
    assert PA.capped_anchor_overflow_ranks(ag, (20, 20, 20)) == (0, 2)
    assert PA.capped_anchor_overflow_ranks((), (20, 20, 20)) == ()
    assert PA.capped_anchor_overflow_ranks(None, None) == ()
    # the wake path uses it for the same drop as the S3 capped guest rows
    from sglang.srt.managers.scheduler_components import weight_updater as WU

    src = inspect.getsource(WU)
    assert src.count("_l15_pa_w.capped_own_ranks(") >= 2 and "_l15_pa_w2.capped_own_ranks(" in src
    # the helper = S3's capped guest ranks UNION the S4b overflow owners
    man = _m4b(ag=((0, 0, 2, 3, 6, 9, 72),), guests=(), caps=(20, 20, 20))
    assert PA.capped_own_ranks(man) == (0,)
    assert PA.capped_own_ranks(_m4b(ag=(), guests=((2, 1, 0, 0, 3),), caps=(20, 20, 20))) == (2,)
    assert PA.capped_own_ranks(_m4b(ag=((0, 1, 0, 2, 4, 3, 24),), guests=(),
                                    caps=(0, 20, 20))) == ()   # cap-0 owner: L2 refill serves
    assert PA.capped_own_ranks(None) == ()


def test_overflow_anchors_of_a_capped_owner_do_not_count_as_the_whole_share_for_the_refill():
    # the L2-skip of the refill (cap-0 rank) needs the owner's pieces to cover EVERY
    # anchor; a capped owner's overflow pieces cover only [cap, n)
    ag = ((0, 0, 2, 3, 6, 9, 72),)
    assert not PA.owner_covers_all_anchors(ag, 0, 5)
    assert PA.owner_covers_all_anchors(((1, 2, 0, 5, 4, 8, 60),), 1, 5)


# -- 7. switches, refusals ----------------------------------------------------------------


def test_switches_default_off_and_named_refusals():
    from sglang.srt.environ import envs

    assert envs.SGLANG_WEG2_L15_POOL_S4B.get() is False
    assert PA.pool_s4b_flag({}) is False and PA.pool_s4b_on({}) is False
    full = {"SGLANG_WEG2_L15_POOL": "1", "SGLANG_WEG2_L15_POOL_S3": "1", PA.POOL_S4_ENV: "1"}
    assert PA.pool_s4b_on({**full, PA.POOL_S4B_ENV: "1"}) is True
    assert PA.pool_s4b_on({"SGLANG_WEG2_L15_POOL": "1", "SGLANG_WEG2_L15_POOL_S3": "1",
                           PA.POOL_S4B_ENV: "1"}) is False          # no S4
    assert PA.pool_s4b_on({PA.POOL_S4_ENV: "1", PA.POOL_S4B_ENV: "1"}) is False
    msg = l15_plan.refuse_pool_s4b_without_s4({PA.POOL_S4B_ENV: "1"})
    assert msg and msg.startswith("W-L15-POOL-S4B-NEEDS-S4") and PA.POOL_S4B_ENV in msg
    two = {"SGLANG_WEG2_L15_POOL": "1", "SGLANG_WEG2_L15_POOL_S3": "1",
           PA.POOL_S4B_ENV: "1"}
    assert l15_plan.refuse_pool_s4b_without_s4(two) is not None      # S4 missing
    assert l15_plan.refuse_pool_s4b_without_s4({**full, PA.POOL_S4B_ENV: "1"}) is None
    assert l15_plan.refuse_pool_s4b_without_s4({}) is None
    env = {PA.POOL_S4B_ENV: "1"}
    msg = l15_plan.refuse_dual(["--dual-layout"], env)
    assert msg and msg.startswith("W-L15-DUAL") and PA.POOL_S4B_ENV in msg
    msg = l15_plan.refuse_not_27b("nextflash", env)
    assert msg and msg.startswith("W-L15-27B-ONLY") and PA.POOL_S4B_ENV in msg
    assert l15_plan.refuse_not_27b("qwen27b", env) is None


def test_launcher_wires_the_s4b_refusal_and_the_boot_line_mode():
    import importlib.util
    import pathlib

    loc = importlib.util.find_spec("sglang.srt.weg2").submodule_search_locations[0]
    src = (pathlib.Path(loc) / "launcher.py").read_text(encoding="utf-8")
    assert "l15_plan.refuse_pool_s4b_without_s4(os.environ)" in src
    assert "_l15_pa4b.s4b_boot_lines(os.environ)" in src
    assert PA.s4b_boot_lines({}) == []
    full = {"SGLANG_WEG2_L15_POOL": "1", "SGLANG_WEG2_L15_POOL_S3": "1", PA.POOL_S4_ENV: "1",
            PA.POOL_S4B_ENV: "1", "SGLANG_WEG2_L15_ANCHOR_CAP": "5"}
    (ln,) = PA.s4b_boot_lines(full)
    assert ln.startswith("L15-POOL-S4B mode=POOL(S4b:") and ln.endswith("anchor_cap=5")
    assert "l15_plan.refuse_pool_s4_without_s3(os.environ)" in src      # S4 as it was
    assert PA.S4B_MODE.startswith("POOL(S4b:")


def test_the_scheduler_resolves_the_cap_through_the_same_pricing_call():
    # the cap rides AnchorCtx, resolved in the one place the S4 pricing is resolved
    import importlib.util
    import pathlib

    loc = importlib.util.find_spec("sglang.srt.managers").submodule_search_locations[0]
    src = (pathlib.Path(loc) / "scheduler.py").read_text(encoding="utf-8")
    assert "_l15_pa.resolve_anchor_ctx(" in src


# -- 8. latent defect found while reading (reported, NOT changed) -----------------------


@pytest.mark.xfail(strict=True, reason=(
    "FINDING (S4b report): l15_share_admit._ratios / l15_deposit_hook._ratios derive the "
    "ANCHOR ratios from d['prefix'] = get_cp_token_ratios() (the DCP TOKEN vector, "
    "published by l15_share_publish) and hand them to MambaBlobSpec.shard_for_rank, "
    "which wants the TP HEAD vector (get_tp_partition_ratios). Today the two differ "
    "-> take_anchor raises L15TakeError 'anchor blob N bytes != spec M'. Remove this "
    "xfail when the anchor path takes the head vector."))
def test_anchor_ratios_of_the_share_path_are_the_head_vector_not_the_token_vector():
    from sglang.srt.weg2 import l15_share_admit

    spec = S4._spec27()
    head = [2, 1, 1]
    token_prefix = [0, 3725, 3725 + 2264, 3725 + 2264 + 2259]
    shares = {0: ({"prefix": token_prefix}, [])}
    got = l15_share_admit._ratios(shares)
    assert got == [3725, 2264, 2259]       # what the code uses today
    want = [spec.shard_for_rank(head, r).total_bytes for r in range(3)]
    have = [spec.shard_for_rank(got, r).total_bytes for r in range(3)]
    assert have == want
