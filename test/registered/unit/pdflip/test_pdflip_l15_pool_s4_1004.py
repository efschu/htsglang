# SPDX-License-Identifier: Apache-2.0
"""L15-POOL stage S4: the END anchors (GDN/Mamba state) in the pool -- a WHOLE
request (KV + anchor) is held or not held at all.

Hermetic: no CUDA, no model, no network.

Run (own worktree, capped):
  cd /spinning/wt-27b-l15-pool-s4-1004 && /spinning/gpu-arb/pytest_gedeckelt.sh \
    test/registered/unit/pdflip/test_pdflip_l15_pool_s4_1004.py

Pinned:
* the anchor share per rank is read from the MambaBlobSpec: NOT equal over the
  ranks (the S1 shadow's ``own-assumed-equal`` was wrong), it follows the TP head
  ratio vector, [2,1,1] gives exactly the 37.4/18.7/18.7 MiB of the boot log, the
  resolver only accepts a ratio candidate the live pool confirms;
* admission counts KV rows AND the anchor bytes of the ranks without a home segment
  against the sum of the segments (mutant: anchors not counted -> a hold without
  anchor room); the exact plan on the COMPACTED rows places the anchors in the free
  hold rows after the KV guests (KV guests = the S3 plan, unchanged), whole anchors
  per piece, once, no overlap; an anchor without room drops the WHOLE request
  (mutant: the half hold, KV without its anchor);
* manifest v2+S4: round trip, the S3 record and fingerprint are byte for byte the
  old ones, the fingerprint covers anchor guests / pricing / row bytes (mutant: not);
* the park moves the anchor shares owner -> host and back byte exact, refuses a
  manifest without anchor fields / another slot width / another row width BEFORE a
  collective, the wake refuses a record that differs from the manifest (anchor
  list), an anchor share that comes back wrong is a group fallback (source
  checksum; mutant: no checksum compare);
* the wake: the scrub spares the rows that host anchors, a refill whose anchors came
  back from the pool loads none from L2 and cannot fall back on a stale L2
  generation (mutant: loads them anyway);
* switches: S4 default off, needs POOL and S3 (refused by name), dual / non-27B
  refuse it; the S3 path is unchanged with S4 off.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import logging
import sys
import types

import pytest
import torch

from flliper.srt.mem_cache.hicache_migrate import qwen3_5_mamba_spec
from flliper.srt.pdflip import l15_manifest as M
from flliper.srt.pdflip import l15_park as P
from flliper.srt.pdflip import l15_plan, l15_pool, l15_pool_anchor as PA, l15_retain

import test_pdflip_l15_park_1002 as T  # _World, _bufs, _entries, _run_ranks
import test_pdflip_l15_pool_s2_1004 as S2  # _plan_inputs
import test_pdflip_l15_cap0_wake_1001 as CW  # the wake fakes

PREFIX = S2.PREFIX
MIB = 1 << 20


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


# -- 1. the anchor share is read from the spec, not assumed equal ---------------------

TC27 = dict(linear_num_value_heads=48, linear_value_head_dim=128, linear_key_head_dim=128,
            linear_num_key_heads=16, linear_conv_kernel_dim=4)


def _spec27():
    return qwen3_5_mamba_spec(TC27, num_linear_layers=48, units=16,
                              temporal_itemsize=2, conv_itemsize=2)


def test_anchor_share_follows_the_head_ratio_vector_and_is_not_equal():
    spec = _spec27()
    assert spec.total_bytes == 78446592                      # 74.8125 MiB per anchor
    two11 = [spec.shard_for_rank([2, 1, 1], r).total_bytes for r in range(3)]
    assert two11 == [39223296, 19611648, 19611648]            # the 2:1:1 of boot 053311
    assert sum(two11) == spec.total_bytes
    assert [spec.shard_for_rank([32, 16, 16], r).total_bytes for r in range(3)] == two11
    even = [spec.shard_for_rank([1, 1, 1], r).total_bytes for r in range(3)]
    assert sum(even) == spec.total_bytes and len(set(even)) == 2      # 6,5,5 units
    auto = [spec.shard_for_rank([3725, 2264, 2259], r).total_bytes for r in range(3)]
    assert len(set(auto)) == 3 and sum(auto) == spec.total_bytes
    # the S1 shadow's assumption "every rank = my own slot bytes" is false in all
    # of them (for TP > 1 there is no vector under which the shares are equal)
    for v in (two11, even, auto):
        assert len(set(v)) > 1


def test_resolver_takes_only_a_ratio_candidate_the_live_pool_confirms():
    spec = _spec27()
    ab, src = PA.anchor_bytes_from_spec(spec, [[1, 1, 1], [2, 1, 1]], 3, 0, 39223296)
    assert ab == (39223296, 19611648, 19611648) and src == "spec(ratios=2,1,1)"
    # rank 1 of the same pool
    ab, _ = PA.anchor_bytes_from_spec(spec, [[1, 1, 1], [2, 1, 1]], 3, 1, 19611648)
    assert ab == (39223296, 19611648, 19611648)
    # nothing confirms this rank's slot: refused by name, never guessed
    ab, why = PA.anchor_bytes_from_spec(spec, [[1, 1, 1], [2, 1, 1]], 3, 0, 12345)
    assert ab is None and "no head-ratio candidate" in why
    ab, why = PA.anchor_bytes_from_spec(spec, [[2, 1, 1]], 3, 0, 0)
    assert ab is None
    # a vector of the wrong length / a zero vector is skipped
    ab, _ = PA.anchor_bytes_from_spec(spec, [[1, 1], [0, 0, 0], [2, 1, 1]], 3, 0, 39223296)
    assert ab == (39223296, 19611648, 19611648)


def test_resolve_anchor_ctx_env_override_and_refusals():
    class V:
        def __init__(self, w):
            self.shape = (8, w)

        def is_contiguous(self):
            return True

    # a sched that has no model: the env route needs only the pool bytes / KV rows
    sched = types.SimpleNamespace()
    env = {PA.l15_pool.POOL_ANCHOR_BYTES_ENV: "24,12,12"}
    import flliper.srt.pdflip.l15_pool as lp

    orig_kv, orig_own = PA._kv_views, lp.own_anchor_slot_bytes
    try:
        PA._kv_views = lambda s: ([V(4), V(4)], None)
        lp.own_anchor_slot_bytes = lambda s: 12
        ctx, why = PA.resolve_anchor_ctx(sched, env, 3, 1)
        assert ctx == PA.AnchorCtx((24, 12, 12), 8, "env") and why == "ok"
        ctx, why = PA.resolve_anchor_ctx(sched, env, 3, 0)          # pool holds 12, env says 24
        assert ctx is None and "holds 12" in why
        ctx, why = PA.resolve_anchor_ctx(
            sched, {PA.l15_pool.POOL_ANCHOR_BYTES_ENV: "24,12"}, 3, 1)
        assert ctx is None and "malformed" in why
        PA._kv_views = lambda s: ([], None)
        ctx, why = PA.resolve_anchor_ctx(sched, env, 3, 1)
        assert ctx is None and "no KV rows" in why
        PA._kv_views = lambda s: (_ for _ in ()).throw(RuntimeError("boom"))
        ctx, why = PA.resolve_anchor_ctx(sched, env, 3, 1)          # never raises
        assert ctx is None and "RuntimeError" in why
    finally:
        PA._kv_views, lp.own_anchor_slot_bytes = orig_kv, orig_own


# -- 2. admission ------------------------------------------------------------------


def _cand(rid, rows, kind="seat"):
    from flliper.srt.pdflip.l15_policy import Candidate

    return Candidate(rid=rid, kind=kind, last_active=0.0, rows_by_rank=tuple(rows),
                     anchor_depth=sum(rows), kv_depth=sum(rows))


def test_s4_admission_counts_the_anchor_bytes_of_the_ranks_without_a_home_segment():
    # caps 0,20,20 (pool 40 rows); row 8 B, anchor share of rank 0 = 24 B = 3 rows
    cands = [_cand("a", (4, 6, 6)), _cand("b", (4, 6, 6)), _cand("c", (4, 6, 6)),
             _cand("d", (1, 1, 0), "parked")]
    s3 = l15_pool.select_hold_pool_s3(cands, [0, 20, 20], 8)
    s4 = PA.select_hold_pool_s4(cands, [0, 20, 20], 8, (24, 12, 12), 8)
    # s3: 16 rows each -> a, b held (32), c needs 16 more (48 > 40), d 2 fits (34)
    assert s3.rids == ("a", "b", "d")
    # s4: each costs 16 + 3 anchor rows: a=19, b=38, c no (54), d: 2 rows fit (40)
    # but not with its 3 anchor rows (43 > 40)
    assert s4.rids == ("a", "b")
    ex = dict(s4.excluded)
    assert ex["c"] == "pool_full" and ex["d"] == "anchor_full"      # kv fits, anchor doesn't
    # no rank without a home segment: the anchor costs the pool nothing, s4 == s3
    assert PA.select_hold_pool_s4(cands, [10, 20, 20], 8, (24, 12, 12), 8).rids == \
        l15_pool.select_hold_pool_s3(cands, [10, 20, 20], 8).rids
    # the anchor count cap, anchorless: today's policy word for word
    many = [_cand("r%d" % i, (1, 1, 1)) for i in range(5)]
    assert len(PA.select_hold_pool_s4(many, [0, 20, 20], 2, (24, 12, 12), 8).rids) == 2
    bad = _cand("x", (1, 1, 1))
    bad = type(bad)(rid="x", kind="seat", last_active=0.0, rows_by_rank=(1, 1, 1),
                    anchor_depth=0, kv_depth=3)
    assert dict(PA.select_hold_pool_s4([bad], [0, 20, 20], 8, (24, 12, 12), 8).excluded) == {
        "x": "anchorless"}


def test_mutant_anchor_does_not_count_into_the_capacity_holds_without_anchor_room():
    cands = [_cand("a", (4, 6, 6)), _cand("b", (4, 6, 6)), _cand("c", (2, 2, 2))]
    caps, ab = [0, 14, 14], (240, 12, 12)                  # 30 anchor rows per anchor
    real = PA.select_hold_pool_s4(cands, caps, 8, ab, 8)
    assert real.rids == ()                                  # 16 + 30 rows > 28
    mut = _mutant("flliper.srt.pdflip.l15_pool_anchor",
                  "        if used + need + a_rows > total_cap:",
                  "        if False:")
    held = mut.select_hold_pool_s4(cands, caps, 8, ab, 8)
    assert held.rids, "the mutant must admit a request whose anchor has no room"
    # and the exact plan refuses what the mutant admitted: the admission is the guard
    _p, _g, why, code = PA.pool_park_plan_s4([4, 6, 6], caps, ab, 8, 1)
    assert why and code == "anchor_full"


# -- 3. the plan -------------------------------------------------------------------


def _check_plan(keep, caps, ab, rb, n, pieces, guests):
    """KV guests + anchor guests: inside the hosts' free rows, no overlap, every
    anchor of every rank without a home segment once, nbytes consistent."""
    R = len(keep)
    home = [min(keep[r], caps[r]) for r in range(R)]
    used = {}
    for p in pieces:
        for i in range(p.rows):
            k = (p.dst, p.dst_row + i)
            assert k not in used
            used[k] = "kv"
    cover = {}
    for (o, h, a_lo, n_a, h_lo, h_rows, nb) in guests:
        assert caps[o] == 0 and o != h and caps[h] > 0 and n_a > 0
        assert nb == n_a * ab[o] and h_rows == -(-nb // rb)
        assert h_lo >= home[h] and h_lo + h_rows <= caps[h]
        for i in range(h_rows):
            k = (h, h_lo + i)
            assert k not in used, "anchor rows overlap %r" % (k,)
            used[k] = "anchor"
        for a in range(a_lo, a_lo + n_a):
            assert (o, a) not in cover
            cover[(o, a)] = 1
    for o in range(R):
        if caps[o] == 0 and ab[o] > 0 and n > 0:
            assert all((o, a) in cover for a in range(n)), (o, cover)


def test_s4_plan_kv_guests_are_the_s3_plan_and_the_anchors_follow_in_the_free_rows():
    keep, caps, ab, rb = [9, 6, 5], [0, 20, 20], (24, 12, 12), 8
    pieces, guests, why, code = PA.pool_park_plan_s4(keep, caps, ab, rb, 3)
    assert why is None and code is None
    want, _ = l15_pool.pool_park_plan(keep, caps)
    assert l15_pool.guest_tuples(pieces) == l15_pool.guest_tuples(want)
    _check_plan(keep, caps, ab, rb, 3, pieces, guests)
    assert sum(g[3] for g in guests) == 3 and all(g[0] == 0 for g in guests)
    # ranks with a home segment keep their share at home: no anchor guest for them
    assert all(g[0] == 0 for g in guests)
    # no rank without a home segment: no anchor guests at all, the S3 plan exactly
    p2, g2, why2, _ = PA.pool_park_plan_s4([9, 6, 5], [10, 20, 20], ab, rb, 3)
    assert g2 == [] and l15_pool.guest_tuples(p2) == l15_pool.guest_tuples(
        l15_pool.pool_park_plan([9, 6, 5], [10, 20, 20])[0])


def test_s4_plan_splits_over_hosts_whole_anchors_and_refuses_by_name():
    # two hosts with 5 free rows each, anchor share 24 B = 3 rows each: one host
    # takes 1 anchor (3 of 5 rows), the next anchors go to the other host
    keep, caps, ab, rb = [0, 5, 5], [0, 12, 12], (24, 12, 12), 8   # 7 free rows each
    pieces, guests, why, _ = PA.pool_park_plan_s4(keep, caps, ab, rb, 3)
    assert why is None, why
    _check_plan(keep, caps, ab, rb, 3, pieces, guests)
    assert len({g[1] for g in guests}) == 2                  # 2 anchors (6 rows) + 1 anchor
    # whole anchors per piece: 5 free rows hold ONE 3-row anchor, never 1.67
    pieces, guests, why, code = PA.pool_park_plan_s4([0, 5, 5], [0, 10, 10], ab, rb, 3)
    assert (pieces, guests) == ([], []) and code == "anchor_full" and "anchor share" in why
    pieces, guests, why, _ = PA.pool_park_plan_s4([0, 5, 5], [0, 10, 10], ab, rb, 2)
    assert why is None and len(guests) == 2 and {g[1] for g in guests} == {1, 2}
    # the KV guests do not fit: the code names pool_full, not anchor_full
    _p, _g, why, code = PA.pool_park_plan_s4([40, 0, 0], [10, 10, 10], ab, rb, 1)
    assert why and code == "pool_full"
    # malformed input is named, never raises
    assert PA.pool_park_plan_s4([1, 2], [0, 1, 2], ab, rb, 1)[2]
    assert PA.pool_park_plan_s4([1, 2, 3], [0, 4, 4], (1, 2), rb, 1)[2]
    assert PA.pool_park_plan_s4([1, 2, 3], [0, 4, 4], ab, 0, 1)[2]


def test_s4_plan_property_valid_cover_or_named_refusal():
    import random

    rnd = random.Random(7)
    for _ in range(400):
        R = rnd.randint(2, 4)
        caps = [rnd.choice([0, 0, rnd.randint(1, 30)]) for _ in range(R)]
        keep = [rnd.randint(0, 25) for _ in range(R)]
        rb = rnd.choice([4, 8, 16])
        ab = tuple(rnd.randint(1, 60) for _ in range(R))
        n = rnd.randint(0, 5)
        pieces, guests, why, code = PA.pool_park_plan_s4(keep, caps, ab, rb, n)
        if why is None:
            _check_plan(keep, caps, ab, rb, n, pieces, guests)
        else:
            assert code in ("pool_full", "anchor_full")
            assert pieces == [] and guests == []
        # the exact plan never needs more than the admission estimated
        if why is None and n:
            est = sum(-(-ab[r] // rb) * n for r in range(R) if caps[r] == 0)
            assert sum(g[5] for g in guests) <= est


def test_s4_plan_is_deterministic_and_rank_uniform():
    a = PA.pool_park_plan_s4([9, 6, 5], [0, 20, 20], (24, 12, 12), 8, 3)
    for _ in range(5):
        assert PA.pool_park_plan_s4([9, 6, 5], [0, 20, 20], (24, 12, 12), 8, 3) == a


def test_q3_the_slowest_card_hosts_anchors_only_as_the_last_overflow():
    # rank 1 is the slowest (low inbound rates): anchors go to rank 2 first
    rates = {(0, 1): 1.0, (2, 1): 1.0, (1, 0): 10.0, (1, 2): 10.0, (0, 2): 10.0, (2, 0): 10.0}
    keep, caps, ab = [0, 0, 0], [0, 30, 30], (24, 12, 12)
    _p, g, why, _ = PA.pool_park_plan_s4(keep, caps, ab, 8, 3, rates)
    assert why is None and {x[1] for x in g} == {2}
    # rank 2 full: the overflow reaches the slow card
    keep, caps = [0, 0, 28], [0, 30, 30]
    _p, g, why, _ = PA.pool_park_plan_s4(keep, caps, ab, 8, 3, rates)
    assert why is None


# -- 4. plan_round -----------------------------------------------------------------

CTX = PA.AnchorCtx((24, 12, 12), 8, "test")


def _plan(caps, n_req=3, s4=True, ctx=CTX, log=None, mod=l15_retain, anchor_slot=5):
    cands, slots = S2._plan_inputs(n_req)
    anchors = {r.rid: i + 1 for i, r in enumerate(cands)} if anchor_slot is None else None
    return mod.plan_round(
        candidates=cands, slots_of=lambda rid: slots[rid],
        anchor_slot_of=(lambda rid: anchor_slot) if anchors is None else
        (lambda rid: 10 + int(str(rid)[-1])),
        caps_rows_by_rank=caps, cap_anchor_slots=8, prefix=PREFIX, epoch=3,
        log=log or (lambda s: None), pool=True, s3=True, rates=None, s4=s4, anchor_ctx=ctx)


def test_s4_off_is_the_s3_plan_byte_for_byte():
    for caps in [(0, 40, 40), (0, 16, 16), (10, 40, 40), (8, 8, 40)]:
        a = _plan(caps, s4=False, ctx=None)
        old = l15_retain.plan_round(
            candidates=S2._plan_inputs()[0], slots_of=lambda rid, s=S2._plan_inputs()[1]: s[rid],
            anchor_slot_of=lambda rid: 5, caps_rows_by_rank=caps, cap_anchor_slots=8,
            prefix=PREFIX, epoch=3, log=lambda s: None, pool=True, s3=True)
        assert (a is None) == (old is None)
        if a is not None:
            assert (a.hs, a.plan, a.guest_pieces, a.pool_fp, a.s3, a.caps) == (
                old.hs, old.plan, old.guest_pieces, old.pool_fp, True, old.caps)
            assert a.anchor_guests is None and a.anchor_ctx is None


def test_s4_plan_holds_whole_requests_with_their_anchor_and_names_it_in_the_line():
    logs = []
    rp = _plan((0, 40, 40), log=logs.append)
    assert rp is not None and rp.anchor_ctx is CTX and rp.anchor_guests
    n = rp.a_h - 1
    rows = list(rp.plan.rows_by_rank)
    kv, ag, why, _c = PA.pool_park_plan_s4(rows, [0, 40, 40], CTX.bytes_by_rank, 8, n)
    assert why is None and tuple(ag) == rp.anchor_guests
    assert tuple(kv) == tuple(rp.guest_pieces)
    _check_plan(rows, [0, 40, 40], CTX.bytes_by_rank, 8, n, kv, ag)
    assert rp.pool_fp == PA.plan_fingerprint_s4(
        rp.hs.rids, rows, (0, 40, 40), kv, ag, CTX.bytes_by_rank, 8, n)
    line = [x for x in logs if x.startswith("L15-POOL-PLAN")][0]
    assert " s3=1 " in line and " s4=1 " in line
    assert "anchor_pieces=%d" % len(ag) in line
    assert "anchor_guest_bytes=%d" % sum(g[6] for g in ag) in line
    assert "anchor_bytes=24,12,12" in line and "anchor_src=test" in line
    assert "anchors=%d " % n in line


def test_s4_without_pricing_the_round_is_not_held_and_says_so():
    logs = []
    assert _plan((0, 40, 40), ctx=None, log=logs.append) is None
    assert any("pool-s4-no-anchor-pricing" in x for x in logs)


def test_s4_a_request_whose_anchor_finds_no_room_is_dropped_whole_never_held_half():
    # the KV rows fit (rank 0 no home: 4 rows/request guest), the anchor shares do not
    big = PA.AnchorCtx((4000, 12, 12), 8, "test")            # 500 rows per anchor
    logs = []
    rp = _plan((0, 40, 40), ctx=big, log=logs.append)
    assert rp is None, "no request has room for its anchor"
    # a pool with room for ONE anchor's rows only: the lowest priority request is dropped
    mid = PA.AnchorCtx((8 * 24, 12, 12), 8, "test")            # 24 rows per anchor
    rp = _plan((0, 40, 40), ctx=mid, log=lambda s: None)
    if rp is not None:
        n = rp.a_h - 1
        rows = list(rp.plan.rows_by_rank)
        kv, ag, why, _ = PA.pool_park_plan_s4(rows, [0, 40, 40], mid.bytes_by_rank, 8, n)
        assert why is None
        _check_plan(rows, [0, 40, 40], mid.bytes_by_rank, 8, n, kv, ag)
        assert len(rp.hs.rids) < 3


def test_mutant_half_hold_kv_without_its_anchor():
    # the half hold: when the anchor shares find no room the plan skips them and
    # keeps the KV -- exactly what 'whole request or not at all' forbids
    mut = _mutant("flliper.srt.pdflip.l15_pool_anchor",
                  "                return [], [], (\"anchor share of rank %d (%d x %d bytes) has no room, \"\n"
                  "                                \"the other segments have %d free rows\"\n"
                  "                                % (owner, left, b, have)), REASON_ANCHOR_FULL\n",
                  "                break\n")
    big = PA.AnchorCtx((4000, 12, 12), 8, "test")
    caps = (0, 40, 40)
    cands, slots = S2._plan_inputs(3)
    rows = [4, 6, 6]
    _p, g, why, _c = mut.pool_park_plan_s4(rows, list(caps), big.bytes_by_rank, 8, 1)
    assert why is None and g == [], "the mutant holds the KV and drops the anchor silently"
    # the real plan refuses the same request
    _p, g, why, code = PA.pool_park_plan_s4(rows, list(caps), big.bytes_by_rank, 8, 1)
    assert why and code == "anchor_full" and g == []


def test_ranks_on_a_different_anchor_pricing_turn_the_round_off_everywhere():
    a = _plan((0, 40, 40), ctx=PA.AnchorCtx((24, 12, 12), 8, "x"))
    b = _plan((0, 40, 40), ctx=PA.AnchorCtx((28, 12, 12), 8, "x"))     # other vector
    assert a.pool_fp != b.pool_fp
    assert l15_pool.agree_pool(None, a.pool_fp, lambda v: [v, (None, b.pool_fp)])
    assert l15_pool.agree_pool(None, a.pool_fp, lambda v: [v, v]) is None


def test_decide_first_passes_the_pricing_to_the_plan():
    from flliper.srt.pdflip import l15_sleep_agree as A

    src = inspect.getsource(A.decide_first)
    assert 'anchor_ctx=kwargs.get("anchor_ctx")' in src


# -- 5. manifest v2 + S4 ---------------------------------------------------------------


def _span(rid="q0"):
    return M.HoldSpan(rid=rid, depth=3, slots=(1, 2, 3), anchor_slot=1, l2_slots=(5, 6, 7),
                      l2_gens=(1, 1, 1), anchor_l2_slot=2, anchor_l2_gen=3, l2_lanes=(0, 1, 0))


def _m4(ag=((0, 1, 0, 2, 4, 3, 24),), ab=(24, 12, 12), rb=8, guests=((0, 1, 2, 4, 2),),
        caps=(2, 10, 10)):
    return M.Manifest(epoch=5, pid=1, spans=(_span(),), rows_by_rank=(12, 5, 6),
                      anchor_slots=3, guests=guests, caps=caps,
                      anchor_guests=ag, anchor_bytes=ab, anchor_row_bytes=rb)


def test_s3_record_and_fingerprint_are_byte_for_byte_the_old_ones():
    # golden values computed with the manifest module of the S3 commit (a46b8ac495)
    m = M.Manifest(epoch=5, pid=1, spans=(_span(),), rows_by_rank=(12, 5, 6), anchor_slots=1,
                   guests=((0, 1, 2, 4, 2), (2, 0, 8, 2, 1)), caps=(2, 10, 10))
    assert m.anchor_guests is None
    assert M.fingerprint(m) == 8753528278516739767
    assert hashlib.sha256(M.to_bytes(m)).hexdigest() == (
        "d29c72abb3535bd6fad03caff843dd4f4f540ac77f430b61c02c923411c7d839")
    assert hashlib.sha256(M.to_json(m).encode()).hexdigest() == (
        "0cc90b12ab9a6ed983986ef6ca68aaecbc25dc7cb41a37733b8a0b210d247ec0")


def test_s4_round_trips_through_bytes_and_json_and_an_s3_record_loads_without_anchors():
    m = _m4()
    for back in (M.from_bytes(M.to_bytes(m)), M.from_json(M.to_json(m))):
        assert back.anchor_guests == m.anchor_guests and back.anchor_bytes == m.anchor_bytes
        assert back.anchor_row_bytes == 8 and back.guests == m.guests and back.caps == m.caps
    s3 = M.Manifest(epoch=5, pid=1, spans=(_span(),), rows_by_rank=(12, 5, 6),
                    anchor_slots=1, guests=((0, 1, 2, 4, 2),), caps=(2, 10, 10))
    back = M.from_bytes(M.to_bytes(s3))
    assert back.anchor_guests is None and back.anchor_bytes is None
    # an empty S4 list is S4 (no rank without a home segment), distinct from S3
    e = _m4(ag=())
    assert M.from_bytes(M.to_bytes(e)).anchor_guests == ()
    assert M.fingerprint(e) != M.fingerprint(s3)
    bad = M.to_json(m).replace('"anchor_guests": [[0, 1, 0, 2, 4, 3, 24]]',
                               '"anchor_guests": [[0, 1, 0]]')
    with pytest.raises(ValueError):
        M.from_json(bad)


def test_fingerprint_covers_anchor_guests_pricing_and_row_bytes():
    base = M.fingerprint(_m4())
    assert M.fingerprint(_m4()) == base
    for other in (_m4(ag=((0, 1, 0, 2, 5, 3, 24),)),       # another host row
                  _m4(ag=((0, 2, 0, 2, 4, 3, 24),)),       # another host
                  _m4(ag=((0, 1, 0, 1, 4, 3, 12),)),       # fewer anchors
                  _m4(ab=(28, 12, 12)),                     # other pricing
                  _m4(rb=16)):                              # other row bytes
        assert M.fingerprint(other) != base


def test_mutant_fingerprint_without_the_anchor_fields_lets_diverged_anchors_agree():
    mut = _mutant("flliper.srt.pdflip.l15_manifest",
                  '    if m.anchor_guests is not None:\n'
                  '        # S4: only an S4 round writes (and hashes) these keys\n',
                  '    if False:\n')
    a, b = _m4(), _m4(ag=((0, 2, 0, 2, 4, 3, 24),))
    assert M.fingerprint(a) != M.fingerprint(b)
    assert mut.fingerprint(a) == mut.fingerprint(b), "the mutant must not see the anchors"


def test_manifest_of_plan_publishes_the_anchor_fields_only_under_s4():
    def man(rp):
        return l15_retain.manifest_of_plan(
            rp, candidates=S2._plan_inputs()[0], l2_of=lambda rid: ((1,), (1,)),
            anchor_l2_of=lambda rid: (2, 3), l2_lanes_of=lambda rid: (0,), epoch=3, pid=1)

    m4 = man(_plan((0, 40, 40)))
    assert m4.anchor_guests == _plan((0, 40, 40)).anchor_guests
    assert m4.anchor_bytes == (24, 12, 12) and m4.anchor_row_bytes == 8
    m3 = man(_plan((0, 40, 40), s4=False, ctx=None))
    assert m3.anchor_guests is None and m3.anchor_bytes is None and m3.guests is not None


# -- 6. the park: anchors owner -> host -> owner --------------------------------------

CAPS = [0, 20, 20]
KEEP = (9, 6, 5)
AB = (24, 12, 12)
RB = 8
N_A = 3
SLOTS = 8


def _s4_env(tmp_path, s4=True):
    env = {"FLLIPER_PDFLIP_L15_PARK_DIR": str(tmp_path), "FLLIPER_PDFLIP_L15_POOL": "1",
           "FLLIPER_PDFLIP_L15_POOL_S3": "1"}
    if s4:
        env["FLLIPER_PDFLIP_L15_POOL_S4"] = "1"
    return env


def _man(keep=KEEP, caps=CAPS, ab=AB, rb=RB, n=N_A, drop=None):
    kv, ag, why, _ = PA.pool_park_plan_s4(list(keep), list(caps), ab, rb, n)
    assert why is None, why
    return M.Manifest(epoch=5, pid=1, spans=(_span(),), rows_by_rank=tuple(keep),
                      anchor_slots=n + 1, guests=l15_pool.guest_tuples(kv),
                      caps=tuple(caps), anchor_guests=PA.anchor_guest_tuples(ag),
                      anchor_bytes=tuple(ab), anchor_row_bytes=rb)


def _mviews(widths=(12, 12), ranks=3):
    """Per-rank fake mamba views: (SLOTS, w) uint8 per view; rank r has widths
    scaled so the share bytes are AB[r] (sum of the widths)."""
    out = {}
    for r in range(ranks):
        ws = (AB[r] // 2, AB[r] - AB[r] // 2)
        views = []
        for v, w in enumerate(ws):
            t = torch.zeros(SLOTS, w, dtype=torch.uint8)
            for s in range(SLOTS):
                for c in range(w):
                    t[s, c] = (31 * r + 17 * v + 7 * s + c + 1) % 251
            views.append(t)
        out[r] = views
    return out


def _wire(monkeypatch, mv):
    monkeypatch.setattr(PA, "mamba_views", lambda sched: mv[sched.rank])


def _expected_rows(mv, g):
    """The host rows the owner's anchors must have become: the shares of the
    anchors back to back, padded, as (host_rows, RB)."""
    o, h, a_lo, n_a, h_lo, h_rows, nb = g
    slots = [1 + a_lo + i for i in range(n_a)]
    flat = torch.cat([torch.cat([v[s] for v in mv[o]]) for s in slots])
    pad = torch.zeros(h_rows * RB, dtype=torch.uint8)
    pad[:nb] = flat
    return pad.view(h_rows, RB)


def _wipe_p_phase(bufs, mv, caps=CAPS, keep=KEEP):
    """What the P phase leaves: the owner's pools (KV + mamba) are garbage; a
    host keeps its hold region (the guest rows)."""
    for b in bufs[0]:
        b.fill_(99)
    for v in mv[0]:
        v.fill_(99)


def test_s4_park_round_trips_the_anchor_shares_byte_exact(monkeypatch, tmp_path):
    m = _man()
    assert m.anchor_guests, "the fixture must have anchor guests"
    mv = _mviews()
    orig = {r: [v.clone() for v in mv[r]] for r in range(3)}
    bufs = {r: T._bufs(r, fill=3 + 40 * r) for r in range(3)}
    korig = {r: [b.clone() for b in bufs[r]] for r in range(3)}
    w, env, scheds = T._entries(monkeypatch, tmp_path, [m, m, m], bufs, CAPS)
    env.update(_s4_env(tmp_path))
    _wire(monkeypatch, mv)
    logs = []
    sent = T._run_ranks(lambda r: P.park_at_release(scheds[r], env, logs.append))
    assert sent[0] > 0 and sent[1] == 0 and sent[2] == 0
    kv_bytes = sum(p.rows for p in l15_pool.pieces_of(m.guests)) * RB
    a_bytes = sum(g[5] for g in m.anchor_guests) * RB
    assert sent[0] == kv_bytes + a_bytes
    # the anchor bytes really lie in the host rows now (stream layout, 2 KV buffers)
    for g in m.anchor_guests:
        exp = _expected_rows(orig, g)
        for L in range(2):
            got = bufs[g[1]][L][g[4]:g[4] + g[5]]
            assert torch.equal(got, exp[:, 4 * L:4 * L + 4])
    # the home rows of the hosts are untouched by the anchors
    for r in (1, 2):
        for L in range(2):
            lo = min(KEEP[r], CAPS[r])
            assert torch.equal(bufs[r][L][:lo], korig[r][L][:lo])
    _wipe_p_phase(bufs, mv)
    back = T._run_ranks(lambda r: P.park_back_at_wake(
        scheds[r], env, logs.append, epoch=5, group_ok=True,
        manifest_guests=m.guests, manifest_anchor_guests=m.anchor_guests))
    assert back == [True, True, True]
    for v, o in zip(mv[0], orig[0]):
        assert torch.equal(v[1:N_A + 1], o[1:N_A + 1])           # the anchors are back
    chk = [x for x in logs if x.startswith("L15-POOL-CHECK")]
    assert len(chk) == 3 and all(" s4=1 anchor_ok=" in x and "anchor_bad=0" in x for x in chk)
    assert any("rank=0" in x and "anchor_pieces=%d" % len(m.anchor_guests) in x for x in chk)
    assert any("rank=1" in x and "anchor_pieces=0" in x for x in chk)
    assert not list(tmp_path.glob("pdflip_l15_park.*"))


def test_s4_park_refuses_before_any_collective(monkeypatch, tmp_path):
    s3_only = M.Manifest(epoch=5, pid=1, spans=(_span(),), rows_by_rank=KEEP, anchor_slots=4,
                         guests=l15_pool.guest_tuples(
                             l15_pool.pool_park_plan(list(KEEP), CAPS)[0]), caps=tuple(CAPS))
    wrong_guests = _man()
    wrong_guests = M.Manifest(**{**wrong_guests.__dict__, "anchor_guests": ((0, 2, 0, 3, 5, 9, 72),)})
    cases = {
        "no anchor fields": (s3_only, "no anchor guests", None),
        "other plan": (wrong_guests, "anchor guest list differs", None),
        "pricing vs pool": (_man(ab=(24, 14, 12)), "slot is", None),
        "row bytes": (_man(rb=16), "KV row is", None),
    }
    for name, (m, needle, _x) in cases.items():
        mv = _mviews()
        bufs = {r: T._bufs(r, fill=3 + 40 * r) for r in range(3)}
        w, env, scheds = T._entries(monkeypatch, tmp_path, [m, m, m], bufs, CAPS)
        env.update(_s4_env(tmp_path))
        _wire(monkeypatch, mv)
        logs = []
        res = T._run_ranks(lambda r: P.park_at_release(scheds[r], env, logs.append))
        assert res == [None] * 3, (name, logs)
        assert w.calls == 0, name
        assert any(needle in x for x in logs), (name, logs)
        assert not list(tmp_path.glob("pdflip_l15_park.*"))


def test_s4_wake_refuses_a_record_that_differs_from_the_manifest_anchor_list(monkeypatch, tmp_path):
    m = _man()
    mv = _mviews()
    bufs = {r: T._bufs(r, fill=3 + 40 * r) for r in range(3)}
    w, env, scheds = T._entries(monkeypatch, tmp_path, [m, m, m], bufs, CAPS)
    env.update(_s4_env(tmp_path))
    _wire(monkeypatch, mv)
    T._run_ranks(lambda r: P.park_at_release(scheds[r], env, lambda s: None))
    other = ((0, 2, 0, 3, 5, 9, 72),)
    logs = []
    back = T._run_ranks(lambda r: P.park_back_at_wake(
        scheds[r], env, logs.append, epoch=5, group_ok=True,
        manifest_guests=m.guests, manifest_anchor_guests=other))
    assert back == [False] * 3
    assert any("differs from the manifest anchor guest list" in x for x in logs)


def _clobber_and_back(monkeypatch, tmp_path, park_mod=P):
    m = _man()
    mv = _mviews()
    bufs = {r: T._bufs(r, fill=3 + 40 * r) for r in range(3)}
    w, env, scheds = T._entries(monkeypatch, tmp_path, [m, m, m], bufs, CAPS)
    env.update(_s4_env(tmp_path))
    _wire(monkeypatch, mv)
    if park_mod is not P:
        for n in ("_group_io", "_kv_buffers", "_caps"):
            monkeypatch.setattr(park_mod, n, getattr(P, n))
    T._run_ranks(lambda r: P.park_at_release(scheds[r], env, lambda s: None))
    g = m.anchor_guests[0]
    for L in range(2):                       # an anchor row clobbered in the host segment
        bufs[g[1]][L][g[4]] = 77
    _wipe_p_phase(bufs, mv)
    logs = []
    back = T._run_ranks(lambda r: park_mod.park_back_at_wake(
        scheds[r], env, logs.append, epoch=5, group_ok=True,
        manifest_guests=m.guests, manifest_anchor_guests=m.anchor_guests))
    return back, logs


def test_s4_an_anchor_share_that_comes_back_wrong_is_a_group_fallback(monkeypatch, tmp_path):
    back, logs = _clobber_and_back(monkeypatch, tmp_path)
    assert back == [False] * 3
    assert any(x.startswith("L15-POOL-CHECK") and "rank=0" in x and "anchor_bad=0" not in x
               for x in logs)


def test_mutant_anchor_guest_without_the_source_checksum_hides_a_clobbered_share(
        monkeypatch, tmp_path):
    # the mutant takes no checksum at the source: the wake has nothing to compare
    monkeypatch.setattr(PA, "anchor_sums_of", lambda guests, rank, m_views: {})
    back, _logs = _clobber_and_back(monkeypatch, tmp_path)
    assert back == [True] * 3, "the mutant should accept the clobbered anchor"


def test_s4_off_the_s3_park_is_unchanged_and_moves_no_anchor(monkeypatch, tmp_path):
    m = _man()
    mv = _mviews()
    bufs = {r: T._bufs(r, fill=3 + 40 * r) for r in range(3)}
    w, env, scheds = T._entries(monkeypatch, tmp_path, [m, m, m], bufs, CAPS)
    env.update(_s4_env(tmp_path, s4=False))
    _wire(monkeypatch, mv)
    logs = []
    sent = T._run_ranks(lambda r: P.park_at_release(scheds[r], env, logs.append))
    kv_bytes = sum(p.rows for p in l15_pool.pieces_of(m.guests)) * RB
    assert sent[0] == kv_bytes
    back = T._run_ranks(lambda r: P.park_back_at_wake(
        scheds[r], env, logs.append, epoch=5, group_ok=True, manifest_guests=m.guests))
    assert back == [True] * 3
    chk = [x for x in logs if x.startswith("L15-POOL-CHECK")]
    assert chk and all(" s4=" not in x for x in chk)


def test_stream_helpers_round_trip_and_the_checksum_sees_every_byte():
    views = [torch.arange(40, dtype=torch.uint8).view(8, 5),
             torch.arange(100, 124, dtype=torch.uint8).view(8, 3)]
    mat = PA.gather_stream(views, [3, 1, 6])
    assert mat.shape == (3, 8)
    assert torch.equal(mat[0], torch.cat([views[0][3], views[1][3]]))
    dst = [torch.zeros(8, 5, dtype=torch.uint8), torch.zeros(8, 3, dtype=torch.uint8)]
    PA.scatter_stream(dst, [3, 1, 6], mat)
    assert torch.equal(dst[0][3], views[0][3]) and torch.equal(dst[1][6], views[1][6])
    with pytest.raises(ValueError):
        PA.scatter_stream(dst, [3, 1], mat)
    flat = torch.arange(PA.SUM_BLOCK + 100, dtype=torch.int64).to(torch.uint8)
    s = PA.stream_sums(flat)
    assert len(s) == 2
    for pos in (0, 7, PA.SUM_BLOCK - 1, PA.SUM_BLOCK + 50):
        f2 = flat.clone()
        f2[pos] ^= 1
        assert PA.stream_sums(f2) != s, pos
    f3 = flat.clone()
    f3[[1, 2]] = flat[[2, 1]]                         # a swap of two bytes
    assert PA.stream_sums(f3) != s


# -- 7. the wake -----------------------------------------------------------------------


def test_wake_scrub_spares_the_rows_that_host_anchors():
    rows = (9, 6, 5)
    guests = ((0, 1, 6, 9, 6),)                       # KV guest rows [6, 15) on rank 1
    ag = ((0, 1, 0, 3, 15, 9, 72), (0, 2, 0, 1, 5, 3, 24))
    assert l15_pool.wake_keep_rows(rows, 1, guests, ag) == 24           # anchors reach row 24
    assert l15_pool.wake_keep_rows(rows, 2, guests, ag) == 8            # rank 2: rows [5,8)
    assert l15_pool.wake_keep_rows(rows, 1, guests, None) == 15         # S3 value, unchanged
    assert l15_pool.wake_keep_rows(rows, 0, guests, ag) == 9
    assert l15_pool.wake_keep_rows(rows, 1, None, ag) == 6              # v1: the old value
    mut = _mutant("flliper.srt.pdflip.l15_pool",
                  "    if anchor_guests:\n        end = max(end,", "    if False:\n        end = max(end,")
    assert mut.wake_keep_rows(rows, 1, guests, ag) == 15, "the mutant scrubs the anchors"
    assert PA.hosted_end_anchor(ag, 1) == 24 and PA.hosted_end_anchor(ag, 0) == 0
    from flliper.srt.managers.scheduler_components import weight_updater as WU

    assert WU._l15_keep_rows_of(_m4(ag=((0, 1, 0, 2, 4, 3, 24),)), 1) == 7


def test_owner_covers_all_anchors():
    g = ((0, 1, 0, 2, 4, 3, 24), (0, 2, 2, 1, 4, 2, 12))
    assert PA.owner_covers_all_anchors(g, 0, 3)
    assert not PA.owner_covers_all_anchors(g, 0, 4)
    assert not PA.owner_covers_all_anchors(g, 1, 3)          # rank 1 owns no piece
    assert not PA.owner_covers_all_anchors((), 0, 0)
    assert not PA.owner_covers_all_anchors(((0, 1, 0, 2, 4, 3, 24),), 0, 3)


def _refill_world(monkeypatch, tmp_path, anchor_gen):
    CW._env(monkeypatch, tmp_path, master=True, mib="c1=64")
    man = CW._fake_manifest(CW._SPANS)
    CW._gate_open(monkeypatch, man)
    host = CW._HostPool({10: 5, 11: 5})
    mamba = CW._MambaHostPool({9: anchor_gen})               # recorded gen is 5
    sched = CW._Sched(host, mamba)
    fs = CW._fake_self(sched, 0)
    WU = CW.WU
    WU._l15_wake_hold_signal(fs)
    WU._pdflip_wake_restore_pools(fs)
    return WU, fs, sched, host, mamba, man


def test_refill_whose_anchors_came_back_from_the_pool_loads_none_from_l2(monkeypatch, tmp_path,
                                                                       caplog):
    # the L2 anchor slot was re-claimed (gen 8 != recorded 5): without the pool the
    # wake would fall back -- the 'Mamba-Anker fehlt' re-prefill; with the anchors
    # back from the pool nothing of L2 is read at all
    WU, fs, sched, host, mamba, man = _refill_world(monkeypatch, tmp_path, anchor_gen=8)
    fs._l15_park_back_ok = True
    fs._l15_anchor_back_ok = True
    with caplog.at_level(logging.INFO, logger=CW.LOGGER):
        n = WU._l15_wake_act(fs, sched, "hold", group_ok=True, master_on=True)
    assert n == 0
    assert mamba.gen_calls == [] and mamba.load_calls == [] and host.load_calls == []
    assert sched.tree_cache.resets == 0                       # no fallback drop
    assert fs._l15_refill_done is True
    assert any("h2d_refill=0 anchors_from_pool=" in r.message for r in caplog.records)
    # the same wake WITHOUT the anchors from the pool (KV parked only) is the fallback
    WU, fs, sched, host, mamba, man = _refill_world(monkeypatch, tmp_path, anchor_gen=8)
    fs._l15_park_back_ok = True
    fs._l15_anchor_back_ok = False
    n = WU._l15_wake_act(fs, sched, "hold", group_ok=True, master_on=True)
    assert n == 0 and sched.tree_cache.resets == 1


def test_mutant_refill_loads_the_anchors_from_l2_although_the_pool_holds_them(
        monkeypatch, tmp_path):
    from flliper.srt.managers.scheduler_components import weight_updater as WU0

    mut = _mutant("flliper.srt.managers.scheduler_components.weight_updater",
                  '            if parked and bool(getattr(self, "_l15_anchor_back_ok", False)):',
                  "            if False:")
    CW._env(monkeypatch, tmp_path, master=True, mib="c1=64")
    man = CW._fake_manifest(CW._SPANS)
    CW._gate_open(monkeypatch, man)
    host = CW._HostPool({10: 5, 11: 5})
    mamba = CW._MambaHostPool({9: 8})
    sched = CW._Sched(host, mamba)
    fs = CW._fake_self(sched, 0)
    fs._l15_do_refill = lambda s: mut.SchedulerWeightUpdaterManager._l15_do_refill(fs, s)
    WU = CW.WU
    WU._l15_wake_hold_signal(fs)
    WU._pdflip_wake_restore_pools(fs)
    fs._l15_park_back_ok = True
    fs._l15_anchor_back_ok = True
    WU._l15_wake_act(fs, sched, "hold", group_ok=True, master_on=True)
    assert mamba.gen_calls, "the mutant must read the stale L2 anchor generation"


# -- 8. retain step 7 --------------------------------------------------------------------


import test_pdflip_l15_retain_0930 as R0  # noqa: E402


def _retain4(tmp_path, monkeypatch, rank, caps, ctx=CTX, s4=True):
    monkeypatch.setenv("FLLIPER_PDFLIP_L15_POOL", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_L15_POOL_S3", "1")
    if s4:
        monkeypatch.setenv("FLLIPER_PDFLIP_L15_POOL_S4", "1")
    else:
        monkeypatch.delenv("FLLIPER_PDFLIP_L15_POOL_S4", raising=False)
    tmp_path.mkdir()
    sc = R0.make_scenario(tmp_path, [])
    sc["kwargs"]["caps_rows_by_rank"] = caps
    sc["kwargs"]["rank"] = rank
    if s4:
        sc["kwargs"]["anchor_ctx"] = ctx
    res = l15_retain.retain_at_sleep(**sc["kwargs"])
    return res, sc, [c[2] for c in sc["set_keep_calls"] if c[0] == "set_keep"]


def test_s4_retain_publishes_the_anchor_guests_and_the_host_keeps_its_whole_region(
        tmp_path, monkeypatch):
    # rows (4, 4), caps (0, 16): rank 0 has no home segment, rank 1 hosts its KV and
    # its anchor shares; the owner keeps NOTHING mapped, the host its whole region
    ctx = PA.AnchorCtx((24, 12), 8, "test")
    res0, sc0, k0 = _retain4(tmp_path / "a", monkeypatch, 0, (0, 16), ctx)
    m = res0.manifest
    assert tuple(m.rows_by_rank) == (4, 4) and m.guests and m.anchor_guests
    assert m.anchor_bytes == (24, 12) and m.anchor_row_bytes == 8
    assert all(g[0] == 0 and g[1] == 1 for g in m.anchor_guests)
    assert m.anchor_slots - 1 == sum(g[3] for g in m.anchor_guests)
    assert all(k == () for k in k0)                          # the owner keeps nothing
    res1, sc1, k1 = _retain4(tmp_path / "b", monkeypatch, 1, (0, 16), ctx)
    assert k1[0] == ((0, 16),)                               # the host: whole hold region
    assert res1.manifest == res0.manifest.__class__(**{**res0.manifest.__dict__, "pid": res1.manifest.pid})
    line = [x for x in sc1["log_lines"] if x.startswith("L15-POOL-PLAN")][0]
    assert " s4=1 " in line and "anchor_pieces=%d" % len(m.anchor_guests) in line
    # S4 off: the S3 record, no anchor field, the S3 window
    res3, sc3, k3 = _retain4(tmp_path / "c", monkeypatch, 1, (0, 16), s4=False)
    assert res3.manifest.anchor_guests is None and res3.manifest.guests is not None


def test_s4_retain_without_pricing_flushes_plain(tmp_path, monkeypatch):
    res, sc, keeps = _retain4(tmp_path / "d", monkeypatch, 0, (0, 16), ctx=None)
    assert res is None and not keeps
    assert any("pool-s4-no-anchor-pricing" in x for x in sc["log_lines"])


# -- 9. switches, refusals ----------------------------------------------------------------


def test_switches_default_off_and_named_refusals():
    from flliper.srt.environ import envs

    assert envs.FLLIPER_PDFLIP_L15_POOL_S4.get() is False
    assert PA.pool_s4_flag({}) is False and PA.pool_s4_on({}) is False
    both = {"FLLIPER_PDFLIP_L15_POOL": "1", "FLLIPER_PDFLIP_L15_POOL_S3": "1"}
    assert PA.pool_s4_on({**both, PA.POOL_S4_ENV: "1"}) is True
    assert PA.pool_s4_on({"FLLIPER_PDFLIP_L15_POOL": "1", PA.POOL_S4_ENV: "1"}) is False   # no S3
    assert PA.pool_s4_on({"FLLIPER_PDFLIP_L15_POOL_S3": "1", PA.POOL_S4_ENV: "1"}) is False
    msg = l15_plan.refuse_pool_s4_without_s3({PA.POOL_S4_ENV: "1"})
    assert msg and msg.startswith("W-L15-POOL-S4-NEEDS-S3") and PA.POOL_S4_ENV in msg
    assert l15_plan.refuse_pool_s4_without_s3(
        {PA.POOL_S4_ENV: "1", "FLLIPER_PDFLIP_L15_POOL": "1"}) is not None
    assert l15_plan.refuse_pool_s4_without_s3({**both, PA.POOL_S4_ENV: "1"}) is None
    assert l15_plan.refuse_pool_s4_without_s3({}) is None
    env = {PA.POOL_S4_ENV: "1"}
    msg = l15_plan.refuse_dual(["--dual-layout"], env)
    assert msg and msg.startswith("W-L15-DUAL") and PA.POOL_S4_ENV in msg
    msg = l15_plan.refuse_not_27b("nextflash", env)
    assert msg and msg.startswith("W-L15-27B-ONLY") and PA.POOL_S4_ENV in msg
    assert l15_plan.refuse_not_27b("qwen27b", env) is None
    assert l15_plan.refuse_dual(["--dual-layout"], {}) is None


def test_launcher_wires_the_s4_refusal_and_the_boot_line_mode():
    import importlib.util
    import pathlib

    loc = importlib.util.find_spec("flliper.srt.pdflip").submodule_search_locations[0]
    src = (pathlib.Path(loc) / "launcher.py").read_text(encoding="utf-8")
    assert "l15_plan.refuse_pool_s4_without_s3(os.environ)" in src
    assert "l15_pool_anchor.pool_s4_on(os.environ)" in src and "L15-POOL-S4 mode=%s" in src
    assert "else l15_pool.S3_MODE))" in src        # the S3 boot line is as it was
    assert PA.S4_MODE.startswith("POOL(S4:")


def test_shadow_prices_the_anchors_per_rank_from_the_spec_not_equal():
    # the shadow's pool side can model the S4 placement (guest anchor in KV rows)
    cands = [_cand("a", (4, 6, 6)), _cand("b", (4, 6, 6))]
    segs = l15_pool.segments_from_caps([0, 20, 20], 8, 8, (24, 12, 12))
    s1 = l15_pool.admit_candidates(cands, segs)                 # S1: anchor slots of the hosts
    s4 = l15_pool.admit_candidates(cands, segs, anchor_in_kv=True)
    assert [p.amount for pl in s1.placements for p in pl.pieces
            if p.kind == "anchor" and not p.home][:1] == [2]    # 2 slots of 12 B
    assert len(s4.admitted) >= 1
    assert all(p.kind != "anchor" or p.home or p.nbytes == 24 for pl in s4.placements
               for p in pl.pieces)
    guest_rows = [p.amount for pl in s4.placements for p in pl.pieces
                  if p.kind == "anchor" and not p.home]
    assert guest_rows and all(a == 3 for a in guest_rows)       # ceil(24 / 8) rows


def test_shadow_line_names_the_anchor_source_never_the_old_equal_assumption(monkeypatch):
    import test_pdflip_l15_pool_1004 as P1

    ctx = PA.AnchorCtx((24, 12, 12), 8, "spec(ratios=2,1,1)")
    monkeypatch.setattr(PA, "resolve_anchor_ctx", lambda s, e, tp, rank: (ctx, "ok"))
    out = []
    env = {"FLLIPER_PDFLIP_L15_MIB": "c1=1,c2=1", "FLLIPER_PDFLIP_L15_ANCHOR_CAP": "2"}
    line = l15_pool.log_sleep_shadow(
        P1._sched(seat=(("s1", 41),)), env, lambda fmt, *a: out.append(fmt % a if a else fmt))
    assert line and "anchor_src=spec(ratios=2,1,1)" in line
    # without a pool-confirmed vector the equal assumption is flagged, never silent
    monkeypatch.setattr(PA, "resolve_anchor_ctx", lambda s, e, tp, rank: (None, "no candidate"))
    sched = P1._sched(seat=(("s1", 41),))
    monkeypatch.setattr(l15_pool, "own_anchor_slot_bytes", lambda s: 4096)
    line = l15_pool.log_sleep_shadow(sched, env, lambda fmt, *a: None)
    assert "anchor_src=own-assumed-equal(UNVERIFIED,wrong-for-tp>1:" in line
