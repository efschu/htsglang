"""Tests for L15-POOL stage S1 (sglang.srt.weg2.l15_pool): the pooled hold
model and its log-only shadow.  Hermetic: no CUDA, no model, no network.

Run (own worktree, capped):
  cd /spinning/wt-27b-l15-pool-s1-1004 && /spinning/gpu-arb/pytest_gedeckelt.sh \
    test/registered/unit/weg2/test_weg2_l15_pool_1004.py

Plain pytest functions only (deterministic pure functions).
"""

import pathlib
import random
import subprocess
import sys
import types

import pytest

from sglang.srt.weg2 import l15_plan, l15_pool
from sglang.srt.weg2.l15_policy import Candidate, select_hold
from sglang.srt.weg2.l15_pool import (
    RequestShards,
    Segment,
    admit_candidates,
    pool_admit,
    pool_capacity,
    segments_from_caps,
    shadow_compare,
    shadow_line,
)

RB = 100  # bytes per KV row in these tests
AB = 1000  # bytes of one anchor slot


def seg(rank, kv, slots=8, row_bytes=RB, slot_bytes=AB):
    return Segment(rank=rank, kv_rows=kv, anchor_slots=slots,
                   row_bytes=row_bytes, anchor_slot_bytes=slot_bytes)


def anchors_for(rids, n, b=AB):
    return {rid: tuple([b] * n) for rid in rids}


def cand(rid, kind, last, rows, anchor_depth=1, kv_depth=1):
    return Candidate(rid=rid, kind=kind, last_active=last, rows_by_rank=tuple(rows),
                     anchor_depth=anchor_depth, kv_depth=kv_depth)


# -- capacity is the SUM -------------------------------------------------------


def test_capacity_is_the_sum_of_the_segments():
    assert pool_capacity([seg(0, 10, 2), seg(1, 0, 0), seg(2, 5, 3)]) == (15, 5)


def test_admission_is_against_the_sum_not_per_rank():
    # rank 0 owns 15 rows but its own segment holds 10: per rank it never
    # fits, against the sum it does (5 rows lie as guests in rank 1).
    segs = [seg(0, 10), seg(1, 10)]
    v = pool_admit([RequestShards("a", (15, 3))], anchors_for(["a"], 2), segs)
    assert v.admitted == ("a",)
    assert v.kv_rows_home_by_rank == (10, 3)
    assert v.kv_rows_guest_by_rank == (5, 0)
    assert v.guest_pairs == {(0, 1): 5 * RB}
    assert v.free_kv_by_rank == (0, 2)
    # today's per-rank policy refuses the same request
    hs = select_hold([cand("a", "seat", 0.0, (15, 3))], [10, 10], 8)
    assert hs.rids == () and hs.excluded == (("a", "no_room"),)


def test_sum_too_small_is_pool_full():
    v = pool_admit([RequestShards("a", (15, 10))], anchors_for(["a"], 2),
                   [seg(0, 10), seg(1, 10)])
    assert v.admitted == () and v.excluded == (("a", "pool_full"),)


# -- the anchor counts into the capacity ---------------------------------------


def test_anchor_slots_are_part_of_the_capacity():
    # KV would fit twice, the anchor slots (1 per rank) hold one request
    segs = [seg(0, 100, slots=1), seg(1, 100, slots=1)]
    v = pool_admit([RequestShards("a", (5, 5)), RequestShards("b", (5, 5))],
                   anchors_for(["a", "b"], 2), segs)
    assert v.admitted == ("a",)
    assert v.excluded == (("b", "anchor_full"),)
    assert v.anchors_home == 2 and v.anchors_guest == 0
    assert v.free_kv_by_rank == (95, 95)  # the refused request took nothing


def test_anchor_guest_goes_to_a_foreign_segment_in_slots_of_the_host():
    # rank 0 has no segment: its head share is 2500 B, the host slot is 1000 B
    # -> ceil(2500 / 1000) = 3 slots on the host
    segs = [seg(0, 0, slots=0), seg(1, 100, slots=4)]
    v = pool_admit([RequestShards("a", (0, 5))], {"a": (2500, 1000)}, segs)
    assert v.admitted == ("a",)
    anchors = [p for p in v.placements[0].pieces if p.kind == "anchor"]
    assert [(p.owner, p.host, p.amount, p.nbytes) for p in anchors] == [
        (0, 1, 3, 2500), (1, 1, 1, 0)]
    assert v.anchors_guest == 1 and v.anchors_home == 1
    assert v.free_anchor_by_rank == (0, 0)
    assert v.guest_bytes_of("anchor") == 2500


def test_anchor_does_not_fit_when_the_host_has_too_few_slots():
    segs = [seg(0, 0, slots=0), seg(1, 100, slots=2)]
    v = pool_admit([RequestShards("a", (0, 5))], {"a": (2500, 1000)}, segs)
    assert v.excluded == (("a", "anchor_full"),)


# -- guest plan: home first -----------------------------------------------------


def test_home_first_then_guest_pieces_in_order():
    segs = [seg(0, 6), seg(1, 100)]
    v = pool_admit([RequestShards("a", (10, 4))], anchors_for(["a"], 2), segs)
    kv = [(p.owner, p.host, p.amount) for p in v.placements[0].pieces if p.kind == "kv"]
    assert kv == [(0, 0, 6), (0, 1, 4), (1, 1, 4)]


def test_a_kv_shard_may_split_over_several_hosts():
    segs = [seg(0, 0, 0), seg(1, 4), seg(2, 4)]
    v = pool_admit([RequestShards("a", (7, 0, 0))], anchors_for(["a"], 3, 0), segs)
    kv = sorted((p.host, p.amount) for p in v.placements[0].pieces if p.kind == "kv")
    assert sum(a for _h, a in kv) == 7 and {h for h, _ in kv} == {1, 2}
    assert v.free_kv_by_rank == (0, 1, 0) or v.free_kv_by_rank == (0, 0, 1)


# -- Q3: free space x rate, the slowest card last --------------------------------


def test_q3_prefers_free_space_times_rate():
    segs = [seg(0, 0, 0), seg(1, 100), seg(2, 40)]
    # rank 1: 100 x 1.0 = 100;  rank 2: 40 x 2.0 = 80  -> rank 1 wins
    v = pool_admit([RequestShards("a", (10, 0, 0))], anchors_for(["a"], 3, 0), segs,
                   rates={(0, 1): 1.0, (0, 2): 2.0})
    assert v.guest_pairs == {(0, 1): 10 * RB}
    # a fast link flips it: rank 2: 40 x 4.0 = 160 > 100
    v = pool_admit([RequestShards("a", (10, 0, 0))], anchors_for(["a"], 3, 0), segs,
                   rates={(0, 1): 1.0, (0, 2): 4.0})
    assert v.guest_pairs == {(0, 2): 10 * RB}


def test_q3_the_slowest_card_takes_only_the_last_overflow():
    # rank 1 is the slow card (inbound x4-like): huge free space, but the
    # guests fill rank 2 first and only the rest goes to rank 1.
    segs = [seg(0, 0, 0), seg(1, 1000), seg(2, 4)]
    rates = {(0, 1): 6.0, (2, 1): 6.0, (0, 2): 13.0, (1, 2): 13.0, (1, 0): 13.0,
             (2, 0): 13.0}
    v = pool_admit([RequestShards("a", (10, 0, 0))], anchors_for(["a"], 3, 0), segs,
                   rates=rates)
    assert v.guest_pairs == {(0, 1): 6 * RB, (0, 2): 4 * RB}


def test_q3_no_rates_means_no_slowest_and_larger_free_wins_then_lower_rank():
    segs = [seg(0, 0, 0), seg(1, 5), seg(2, 5), seg(3, 9)]
    v = pool_admit([RequestShards("a", (4, 0, 0, 0))], anchors_for(["a"], 4, 0), segs)
    assert v.guest_pairs == {(0, 3): 4 * RB}
    v = pool_admit([RequestShards("a", (4, 0, 0, 0))], anchors_for(["a"], 4, 0),
                   [seg(0, 0, 0), seg(1, 5), seg(2, 5)])
    assert v.guest_pairs == {(0, 1): 4 * RB}  # tie -> lower rank


# -- a rank without a segment ----------------------------------------------------


def test_rank_without_segment_is_all_guest_and_does_not_block():
    segs = [seg(0, 0, 0), seg(1, 100), seg(2, 100)]
    v = pool_admit([RequestShards("a", (6, 3, 3))], anchors_for(["a"], 3), segs)
    assert v.admitted == ("a",)
    assert v.kv_rows_home_by_rank == (0, 3, 3)
    assert v.kv_rows_guest_by_rank == (6, 0, 0)
    assert v.anchors_guest == 1


def test_rank_missing_from_the_segment_list_is_the_same():
    v = pool_admit([RequestShards("a", (6, 3))], anchors_for(["a"], 2),
                   [seg(1, 100)])
    assert v.admitted == ("a",)
    assert v.kv_rows_guest_by_rank == (6, 0)
    # row bytes of a rank without a segment come from the pool's row size
    assert v.guest_bytes_of("kv") == 6 * RB


# -- edge cases ------------------------------------------------------------------


def test_zero_space_everywhere_holds_nothing():
    v = pool_admit([RequestShards("a", (1, 0))], anchors_for(["a"], 2),
                   [seg(0, 0, 0), seg(1, 0, 0)])
    assert v.admitted == () and v.excluded == (("a", "pool_full"),)
    assert v.capacity_kv_rows == 0 and v.guest_bytes == 0


def test_no_kv_but_anchor_with_zero_slots_is_anchor_full():
    v = pool_admit([RequestShards("a", (0, 0))], anchors_for(["a"], 2),
                   [seg(0, 5, 0), seg(1, 5, 0)])
    assert v.excluded == (("a", "anchor_full"),)


def test_empty_inputs_and_zero_rows_zero_anchor():
    assert pool_admit([], {}, []).admitted == ()
    v = pool_admit([RequestShards("a", (0, 0))], {"a": (0, 0)}, [seg(0, 0, 0)])
    assert v.admitted == ("a",) and v.placements[0].pieces == ()


def test_all_or_nothing_and_a_later_smaller_request_still_fits():
    segs = [seg(0, 10, 8), seg(1, 10, 8)]
    v = pool_admit(
        [RequestShards("big", (30, 0)), RequestShards("small", (4, 4))],
        anchors_for(["big", "small"], 2), segs)
    assert v.admitted == ("small",)
    assert v.excluded == (("big", "pool_full"),)
    assert v.free_kv_by_rank == (6, 6)


def test_anchorless_when_missing_or_none():
    v = pool_admit([RequestShards("a", (1, 1)), RequestShards("b", (1, 1))],
                   {"b": None}, [seg(0, 10), seg(1, 10)])
    assert v.admitted == () and set(v.excluded) == {("a", "anchorless"),
                                                    ("b", "anchorless")}
    assert v.excluded_counts()["anchorless"] == 2


def test_malformed_input_raises_value_error():
    with pytest.raises(ValueError):
        pool_admit([RequestShards("a", (-1, 0))], {"a": (1, 1)}, [seg(0, 1)])
    with pytest.raises(ValueError):
        pool_admit([RequestShards("a", (1,)), RequestShards("a", (1,))],
                   {"a": (1,)}, [seg(0, 5)])
    with pytest.raises(ValueError):
        pool_admit([], {}, [seg(0, 1), seg(0, 2)])


# -- invariants (fuzz) -------------------------------------------------------------


def test_fuzz_no_host_is_overcommitted_and_bytes_add_up():
    rng = random.Random(1004)
    for _ in range(300):
        n = rng.randint(1, 4)
        segs = [seg(r, rng.choice([0, 3, 10, 40]), rng.choice([0, 1, 3]),
                    slot_bytes=rng.choice([500, 1000])) for r in range(n)]
        rids = [f"r{i}" for i in range(rng.randint(0, 8))]
        reqs = [RequestShards(rid, tuple(rng.randint(0, 15) for _ in range(n)))
                for rid in rids]
        anc = {rid: tuple(rng.choice([0, 700, 1000]) for _ in range(n)) for rid in rids}
        rates = {(a, b): rng.choice([1.0, 6.0, 13.0]) for a in range(n) for b in range(n)
                 if a != b} if rng.random() < 0.5 else None
        v = pool_admit(reqs, anc, segs, rates=rates)
        used_kv = [0] * n
        used_a = [0] * n
        for pl in v.placements:
            for p in pl.pieces:
                (used_kv if p.kind == "kv" else used_a)[p.host] += p.amount
        for r, s in enumerate(segs):
            assert used_kv[r] <= s.kv_rows and used_a[r] <= s.anchor_slots
            assert v.free_kv_by_rank[r] == s.kv_rows - used_kv[r]
            assert v.free_anchor_by_rank[r] == s.anchor_slots - used_a[r]
        # every admitted request is whole: all its KV rows and anchor shares placed
        for pl in v.placements:
            req = next(q for q in reqs if q.rid == pl.rid)
            for r, rows in enumerate(req.rows_by_rank):
                assert sum(p.amount for p in pl.pieces
                           if p.kind == "kv" and p.owner == r) == rows
            for r, b in enumerate(anc[pl.rid]):
                assert sum(1 for p in pl.pieces if p.kind == "anchor" and p.owner == r) \
                    == (1 if b > 0 else 0)
        assert len(v.admitted) + len(v.excluded) == len(reqs)
        assert sum(v.guest_pairs.values()) == v.guest_bytes


def test_fuzz_pool_is_deterministic_and_independent_of_candidate_input_order():
    rng = random.Random(7)
    for _ in range(100):
        n = 3
        segs = segments_from_caps([rng.choice([0, 20, 50]) for _ in range(n)], 3, RB,
                                  [AB] * n)
        cands = [cand(f"c{i}", rng.choice(["seat", "parked", "served"]),
                      float(rng.randint(0, 5)), [rng.randint(0, 12) for _ in range(n)],
                      1, 1 if rng.random() < 0.9 else 2) for i in range(8)]
        shuffled = cands[:]
        rng.shuffle(shuffled)
        a = admit_candidates(cands, segs)
        b = admit_candidates(shuffled, segs)
        assert a.admitted == b.admitted and a.guest_pairs == b.guest_pairs
        assert set(a.excluded) == set(b.excluded)


# -- the S1 comparison with today's path (golden) ---------------------------------


def test_shadow_compare_today_side_is_exactly_select_hold():
    rng = random.Random(930)
    for _ in range(100):
        n = rng.randint(1, 3)
        caps = [rng.choice([0, 10, 60]) for _ in range(n)]
        cands = [cand(f"c{i}", rng.choice(["seat", "parked", "served"]),
                      float(rng.randint(0, 4)), [rng.randint(0, 9) for _ in range(n)],
                      1, 1 if rng.random() < 0.85 else 3) for i in range(rng.randint(0, 7))]
        ps = shadow_compare(cands, caps, 4, row_bytes=RB,
                            anchor_slot_bytes_by_rank=[AB] * n)
        assert ps.today == select_hold(cands, caps, 4)  # Byte fuer Byte


def test_pool_holds_more_whole_requests_than_the_per_rank_path_when_a_rank_is_full():
    # rank 1's segment is small; per rank the 2nd request is "no_room", the
    # pool places the overflow as guests in rank 0.
    cands = [cand("a", "seat", 2.0, (3, 8)), cand("b", "seat", 1.0, (3, 8))]
    ps = shadow_compare(cands, [100, 10], 8, row_bytes=RB,
                        anchor_slot_bytes_by_rank=[AB, AB])
    assert ps.today.rids == ("a",)
    assert ps.pool.admitted == ("a", "b")
    assert ps.pool_only == ("b",) and ps.today_only == ()
    assert ps.pool.kv_rows_guest_by_rank == (0, 6)
    assert ps.pool.guest_pairs == {(1, 0): 6 * RB}


def test_pool_can_hold_less_when_the_cap0_rank_needs_guest_room_and_anchors():
    # today's policy skips the cap-0 rank completely (its rows and anchor come
    # from L2); the pool must find room for them -> a visible difference.
    cands = [cand("a", "seat", 1.0, (50, 4))]
    ps = shadow_compare(cands, [0, 20], 8, row_bytes=RB,
                        anchor_slot_bytes_by_rank=[AB, AB])
    assert ps.today.rids == ("a",)
    assert ps.pool.admitted == () and ps.today_only == ("a",)
    assert ps.pool.excluded == (("a", "pool_full"),)


def test_segments_from_caps_gives_a_cap0_rank_no_anchor_region():
    segs = segments_from_caps([0, 7], 8, RB, [AB, AB], cards=[1, 2])
    assert [(s.kv_rows, s.anchor_slots, s.card) for s in segs] == [(0, 0, 1), (7, 8, 2)]


def test_shadow_line_fields():
    cands = [cand("a", "seat", 2.0, (3, 8)), cand("b", "seat", 1.0, (3, 8)),
             cand("x", "served", 0.0, (1, 1), 1, 2)]
    ps = shadow_compare(cands, [100, 10], 8, row_bytes=RB,
                        anchor_slot_bytes_by_rank=[AB, AB])
    line = shadow_line(7, ps, [100, 10], anchor_src="env")
    assert line.startswith("L15-POOL-SHADOW at=sleep epoch=7 req=3 ")
    for frag in ("today_n=1 ", "pool_n=2 ", "pool_tokens=22 ", "pool_only=1 ",
                 "kv_rows_guest=0,6 ", "guest_pairs=1>0:600 ",
                 "excluded=pool_full:0,anchor_full:0,anchorless:1 ",
                 "cap_rows=100,10 ", "cap_sum_rows=110 ", "anchor_src=env"):
        assert frag in line, (frag, line)


# -- switch, boot line, parsers, refusals -----------------------------------------


def test_switch_is_off_by_default_and_parses_on_values():
    assert l15_pool.pool_shadow_on({}) is False
    assert l15_pool.pool_shadow_on({l15_pool.POOL_SHADOW_ENV: "0"}) is False
    for v in ("1", "true", "ON", " yes "):
        assert l15_pool.pool_shadow_on({l15_pool.POOL_SHADOW_ENV: v}) is True


def test_environ_declares_the_switch_default_false():
    from sglang.srt.environ import envs

    assert envs.SGLANG_WEG2_L15_POOL_SHADOW.default is False


def test_boot_line_is_the_sum_of_the_posts():
    line = l15_pool.boot_line([0, 7616, 1792], ["OVERRIDE-UNNAMED", "OVERRIDE", "OVERRIDE"], 8)
    assert line.startswith("L15-POOL cards=3 segments=2 mib=0,7616,1792 total_mib=9408 anchor_cap=8 ")
    assert "src=OVERRIDE,OVERRIDE-UNNAMED" in line and "no behaviour change" in line


def test_parsers_never_raise_and_reject_malformed():
    assert l15_pool.parse_rates("0>1=13.4, 1>0=6") == {(0, 1): 13.4, (1, 0): 6.0}
    for bad in (None, "", "0>1", "0>1=x", "a>b=1", "0>1=-2", "0-1=3"):
        assert l15_pool.parse_rates(bad) is None
    assert l15_pool.parse_int_list("1, 2,3") == (1, 2, 3)
    for bad in (None, "", "1,x", "-1,2"):
        assert l15_pool.parse_int_list(bad) is None


def test_dual_and_not27b_refuse_the_pool_shadow_switch_by_name():
    env = {l15_pool.POOL_SHADOW_ENV: "1"}
    msg = l15_plan.refuse_dual(["--dual-layout"], env)
    assert msg and msg.startswith("W-L15-DUAL") and l15_pool.POOL_SHADOW_ENV in msg
    msg = l15_plan.refuse_not_27b("nextflash", env)
    assert msg and msg.startswith("W-L15-27B-ONLY") and l15_pool.POOL_SHADOW_ENV in msg
    # switch off: both refusals are exactly today's (nothing armed -> None)
    assert l15_plan.refuse_dual(["--dual-layout"], {}) is None
    assert l15_plan.refuse_not_27b("nextflash", {}) is None
    assert l15_plan.refuse_not_27b("qwen27b", env) is None


def test_module_is_torch_free():
    # the package import pulls torch in this tree, so check the module's own
    # import graph: l15_pool and l15_policy import stdlib only (the helpers that
    # need torch-bound modules import them lazily, inside functions).
    import ast
    import importlib.util

    for name in ("sglang.srt.weg2.l15_pool", "sglang.srt.weg2.l15_policy"):
        path = pathlib.Path(importlib.util.find_spec(name).origin)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        mods = set()
        for n in top:
            mods.update([n.module] if isinstance(n, ast.ImportFrom) else
                        [a.name for a in n.names])
        assert not any(m and m.split(".")[0] in ("torch", "numpy") for m in mods), mods
        assert all(m.split(".")[0] in ("__future__", "dataclasses", "typing", "sglang")
                   for m in mods if m), mods


# -- the scheduler hook ---------------------------------------------------------------


class _T:  # a fake KV tensor: (rows, heads, dim) bf16
    shape = (100, 8, 16)

    def numel(self):
        return 100 * 8 * 16

    def element_size(self):
        return 2

    def data_ptr(self):
        return 0


def _sched(tp=3, seat=(("s1", 41),), parked=()):
    def req(rid, tok):
        return types.SimpleNamespace(rid=rid, origin_input_ids=[0] * tok, output_ids=[])

    pool = types.SimpleNamespace(k_buffer=[_T()], v_buffer=[_T()])
    return types.SimpleNamespace(
        tp_size=tp,
        tp_worker=types.SimpleNamespace(
            model_runner=types.SimpleNamespace(token_to_kv_pool=pool)),
        server_args=types.SimpleNamespace(tp_size=tp, rank_gpu_id=None),
        running_batch=types.SimpleNamespace(reqs=[req(r, t) for r, t in seat]),
        weg2_d_parked=[req(r, t) for r, t in parked],
        req_to_token_pool=None,
    )


def test_log_sleep_shadow_end_to_end_with_fake_scheduler():
    out = []
    env = {"SGLANG_WEG2_L15_MIB": "c1=1,c2=1", "SGLANG_WEG2_L15_ANCHOR_CAP": "2"}
    line = l15_pool.log_sleep_shadow(
        _sched(seat=(("s1", 41), ("s2", 4001)), parked=(("p1", 21),)), env,
        lambda fmt, *a: out.append(fmt % a if a else fmt))
    assert line is not None and out == [line]
    # 1 MiB / (8*16*2*2 = 512 B per row) = 2048 rows on ranks 1 and 2, rank 0 cap 0
    assert "cap_rows=0,2048,2048 " in line
    assert "anchor_cap=2 " in line and "anchor_src=unit-slots(bytes unknown)" in line
    assert "req=3 " in line and "today_n=2 " in line


def test_log_sleep_shadow_never_raises():
    # a log function that raises is swallowed
    def boom(*_a):
        raise RuntimeError("log down")

    assert l15_pool.log_sleep_shadow(_sched(), {"SGLANG_WEG2_L15_MIB": "c1=1"}, boom) is None


def test_scheduler_hook_is_guarded_and_after_the_old_shadow_block():
    import importlib.util

    _loc = importlib.util.find_spec("sglang.srt.managers").submodule_search_locations[0]
    src = (pathlib.Path(_loc) / "scheduler.py").read_text(encoding="utf-8")
    i_old = src.index('logger.warning("L15-SHADOW select failed (ignored)')
    i_new = src.index("_l15_pool_ps.pool_shadow_on(os.environ)")
    assert i_new > i_old
    block = src[i_new - 400:i_new + 700]
    assert "_l15_plan_ps.master_on(" in block and "log_sleep_shadow(self, os.environ" in block
    assert "except Exception" in block  # never raises into the flush
