"""#239 S3f: the planner sets the MoE ownership under the token cut.

Target form (Nutzer 28.09.): the full-attention KV lies on the 3080 workers,
the attention host (5090) holds none of it (share 0; QSA index, Mamba and
draft stay), and the expert ownership (``--rank-moe-ratio``) moves to the host
card the KV frees -- a fixed part of the form, set by the planner, never a
hand value (the S4b profile nf-s4a-cut.env carried 215/117/156 and 0,48,16 by
hand). S2b's joint solve kept the ownership at 183/137/168 and maximised the
smallest resident share; its cut 0/46/18 was refused by W130.

``--d-kv-token-cut owned`` solves ownership, the workers' shares and FR_D in
one pass: min max_r of the per-rank miss time (rows the card cannot hold x
MoE layers x cost per row, seed 0.1 / 0.2 ms, UNMEASURED until M1), with the
x1 rule -- no worker misses more than in Form A.
"""

from __future__ import annotations

import inspect
import types

import pytest

from sglang.srt.planner import expert_residency as er
from sglang.srt.weg2 import launcher as L

BASE = (183, 137, 168)
CAP = (200, 150, 170)       # rows the budget carries before KV, per rank
SCRATCH = (51, 48, 40)
KV_ROWS = 53                # the full-attention KV at 524k, in rows


def _fit(rank, ratio, share, cut):
    E = int(ratio) + 10
    if cut is None:
        kv = KV_ROWS if rank == 0 else 0            # Form A: the host holds all
    else:
        kv = KV_ROWS * cut[rank] / float(sum(cut))
    ceiling = int(CAP[rank] - kv)
    S = SCRATCH[rank]
    rows = min(ceiling - S, E - 2)
    return types.SimpleNamespace(
        rank=rank, local_experts=E, ceiling_max_rows=ceiling, scratch_rows=S,
        ceiling_fraction=(rows / float(E)) if rows >= 1 else None)


def _solve_at(rat, sh):
    return tuple(_fit(r, rat[r], None, sh) for r in range(3))


def _owned(**kw):
    return er.solve_owned_cut(_solve_at, BASE, 0, num_experts=512, n_layers=48,
                              ids_per_step=40, **kw)


def test_the_host_holds_no_kv_and_takes_ownership():
    sol = _owned()
    assert sol.feasible > 0
    assert sol.cut[0] == 0 and sum(sol.cut) == er.KV_TOKEN_SHARE_GRID
    # ownership moved to the host, the sum is kept
    assert sol.ratios[0] > BASE[0] and sum(sol.ratios) == sum(BASE)
    # FR_D is every rank's edge at the chosen form
    assert all(f > 0 for f in sol.fractions)


def test_the_x1_rule_no_worker_misses_more_than_form_a():
    sol = _owned()
    assert all(sol.round_ms[w] <= sol.base_round_ms[w] + 1e-9 for w in (1, 2))
    # the target form is not promised to beat Form A at its worst rank (the
    # host takes the misses the workers give up); the solve reports both and
    # M1 measures -- no silent switch back (main 28.09.)
    assert len(sol.base_round_ms) == 3 and max(sol.round_ms) > 0


def test_no_worker_room_means_no_form():
    global CAP
    keep = CAP
    try:
        CAP = (200, 60, 60)       # the workers cannot carry the KV at all
        sol = _owned()
    finally:
        CAP = keep
    assert sol.feasible == 0 and sol.ratios == () and sol.cut == ()
    assert sol.candidates > 0


def test_ratio_vectors_keep_the_sum_and_start_at_the_stated_one():
    vecs = er.owned_ratio_vectors(BASE, 0, step=8, max_shift=16)
    assert vecs[0] == BASE
    assert all(sum(v) == sum(BASE) for v in vecs)
    assert (199, 129, 160) in vecs and (199, 137, 152) in vecs
    assert all(v[0] >= BASE[0] for v in vecs)


def test_the_miss_cost_is_per_card():
    host = types.SimpleNamespace(rank=0, local_experts=200, ceiling_max_rows=100)
    worker = types.SimpleNamespace(rank=1, local_experts=200, ceiling_max_rows=100)
    a, b = er.owned_round_ms((host, worker), host=0, num_experts=512, ids_per_step=40,
                             n_layers=48)
    assert b == pytest.approx(2 * a)       # 0.2 vs 0.1 ms per row (seed)


# ---- the launcher -------------------------------------------------------------

def test_the_flag_accepts_owned():
    ns = types.SimpleNamespace(d_kv_token_cut="owned")
    assert L.d_kv_token_cut(ns) == er.KV_TOKEN_CUT_OWNED


def test_publish_writes_the_ownership_and_keeps_the_stated_vector(capsys):
    ns = types.SimpleNamespace(extra_d="--rank-tp-ratio 1,0,0 --rank-moe-ratio 183,137,168")
    plan = types.SimpleNamespace(solved_owner_ratio=(215, 117, 156),
                                 owner_record=(("cut", [0, 48, 16]),))
    lines = []
    out = L.publish_d_owner_ratio(ns, plan, "D", lines.append)
    assert out == [215.0, 117.0, 156.0]
    assert "--rank-moe-ratio 215,117,156" in ns.extra_d
    assert "183,137,168" not in ns.extra_d
    assert ns._d_owner_solve["stated_ratios"] == ["183", "137", "168"]
    assert ns._d_owner_solve["cut"] == [0, 48, 16]
    assert lines and "D-EIGENTUM (#239 S3f) veroeffentlicht" in lines[0]
    # a second pass (the real run after the expectation) keeps the STATED one
    plan2 = types.SimpleNamespace(solved_owner_ratio=(223, 113, 152), owner_record=())
    L.publish_d_owner_ratio(ns, plan2, "D", lines.append)
    assert ns._d_owner_solve["stated_ratios"] == ["183", "137", "168"]
    assert "--rank-moe-ratio 223,113,152" in ns.extra_d


def test_the_solve_starts_every_pass_from_the_stated_vector():
    src = inspect.getsource(L.log_d_rank_vram_solve)
    i = src.index("_kv_cut = d_kv_token_cut(ns)")
    j = src.index("ratios = list(ns._d_owner_stated)", i)
    assert j - i < 300
    assert src.index("publish_d_owner_ratio(ns, plan, label, log)") < src.index(
        "if plan.solved_fractions:")


def test_the_record_carries_the_ownership_solve():
    assert "d_owner_solve" in L.BootState.__dataclass_fields__
    main = inspect.getsource(L.main)
    assert 'state.d_owner_solve = dict(getattr(ns, "_d_owner_solve", None) or {})' in main


def test_plan_d_residency_knows_owned():
    src = inspect.getsource(er.plan_d_residency)
    assert "kv_token_shares == KV_TOKEN_CUT_OWNED" in src
    assert "solved_owner_ratio=solved_owner" in src
    assert "owner_refusal" in src


def test_the_card_edge_bounds_the_solve():
    # the budget alone would let the host hold more rows than its CARD
    # carries (W130 then refused the plan in the desk probe at 262k); the
    # solve takes min(budget, card) as the edge
    free = _owned()
    capped = er.solve_owned_cut(_solve_at, BASE, 0, num_experts=512, n_layers=48,
                                ids_per_step=40, card_rows=lambda fits: (120, 999, 999))
    assert capped.feasible > 0
    assert capped.fractions[0] <= free.fractions[0]
    E0 = capped.ratios[0] + 10
    assert capped.fractions[0] <= (120 - SCRATCH[0]) / float(E0) + 1e-9


# ---- the attention and LSE posts (main 28.09.: named, seed UNMEASURED) --------

def test_the_attention_post_follows_the_share_and_the_merge_is_on_every_rank():
    f = [types.SimpleNamespace(rank=r, local_experts=100, ceiling_max_rows=100)
         for r in range(3)]            # no misses: only the new posts remain
    kw = dict(host=0, num_experts=512, ids_per_step=40, n_layers=48, fa_layers=12,
              rows_per_round=4)
    split = er.owned_round_ms(f, shares=(0, 32, 32), merged=True, **kw)
    one = er.owned_round_ms(f, shares=(0, 64, 0), merged=True, **kw)
    lse = 12 * er.OWNED_LSE_MS_PER_LAYER_SEED
    floor = 12 * er.OWNED_ATTN_FLOOR_MS_SEED
    row = er.OWNED_ATTN_MS_PER_ROW_SEED[1]
    # the host holds no KV: only the merge
    assert split[0] == pytest.approx(lse) and one[0] == pytest.approx(lse)
    # a worker pays floor + rows x share x cost, plus the merge
    assert split[1] == pytest.approx(lse + floor + 12 * 4 * 0.5 * row)
    assert one[1] == pytest.approx(lse + floor + 12 * 4 * 1.0 * row)
    assert one[2] == pytest.approx(lse)
    # Form A: the host attends over the whole KV, no merge
    base = er.owned_round_ms(f, shares=(1, 0, 0), merged=False, **kw)
    assert base[0] == pytest.approx(floor + 12 * 4 * er.OWNED_ATTN_MS_PER_ROW_SEED[0])
    assert base[1] == 0.0


def test_the_solve_and_the_record_carry_the_attention_post():
    src = inspect.getsource(er.solve_owned_cut)
    assert "owned_round_ms(fits, shares=sh, merged=True, **kw)" in src
    assert "merged=False" in src
    plan_src = inspect.getsource(er.plan_d_residency)
    for key in ('"attn_ms_per_row"', '"lse_ms_per_layer"', '"attn_source"', '"fa_layers"'):
        assert key in plan_src
    assert "rows_per_round=int(verify)" in plan_src
