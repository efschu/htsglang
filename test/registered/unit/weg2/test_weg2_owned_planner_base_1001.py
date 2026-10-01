"""Owned cut from the profiles (user 01.10. ~19:05Z: the expert split on D
comes from the PLANNER, derived from the hardware and model profile, not a
hand vector in the profile).

Four inputs of ``solve_owned_cut`` were wrong on the y6k -dres line
(audit 01.10.):

* the base was the profile's hand vector (183,137,168) and the x1 guard held
  every worker to THAT form -- TP1 sat at its x1 edge, nothing could leave
  the 5090;
* the cost per missed row was one pooled worker cost (0.314 ms) for two
  3080s on different links: TP1 = NVML0 x4 0.473, TP2 = NVML2 x8 0.217 ms
  (rank records, 6-h window to 20261001T185830);
* the objective was bs1 alone while D ran bs2/bs3 in 89 % of its steady
  rounds (bs1 513 / bs2 2143 / bs3 2160);
* no heat entered the non-resident demand.

The edge model below is the live launcher's (18:23:32Z, lines 175-177 of
boot ...dauer10011823_a11cc7cbc6_1001_182321.front.log); the old solve on it
reproduces the published 215,121,152 with T_r [68.78, 13.33, 41.64] to the
hundredth, so the new numbers are the planner's on the same edge.
"""

from __future__ import annotations

import inspect
import types

import pytest

from sglang.srt.planner import expert_residency as er
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-test-cpu")

NUM_E = 512
N_LAYERS = 48
FA = 12
VERIFY = 4
IDS = 40
ROW_MIB = 48 * 2.417
A = (15430 + 541, 13574 + 1967, 14386 + 1157)  # Puffer + Rest vor KV, MiB
HOST_REST_MIB = 464
FA_MIB = 3072
S = (104, 48, 48)
HAND = (183, 137, 168)
POOLED = (0.201074, 0.314024)
PER_RANK = (0.2027, 0.4734, 0.2171)
KW = dict(num_experts=NUM_E, n_layers=N_LAYERS, ids_per_step=IDS, fa_layers=FA,
          rows_per_round=VERIFY)


def solve_at(rat, sh):
    spans = er.expert_span_by_rank(num_experts=NUM_E, ratios=list(rat))
    out = []
    for r in range(3):
        if sh is None:
            kv = (HOST_REST_MIB + FA_MIB) if r == 0 else 0
        else:
            kv = (HOST_REST_MIB if r == 0 else 0) + FA_MIB * sh[r] / 64.0
        rows = int((A[r] - kv) // ROW_MIB)
        E = int(spans[r]) + 1
        frac = er.largest_fraction_for_rows(local_experts=E, scratch_rows=S[r], max_rows=rows)
        out.append(types.SimpleNamespace(rank=r, local_experts=E, scratch_rows=S[r],
                                         ceiling_max_rows=rows, ceiling_fraction=frac))
    return tuple(out)


def _rec(rank, fetch_ms, rows, rounds=40, t=1000.0):
    return {"kind": er.OWNED_MISS_RANK_KIND, "pairing": er.OWNED_MISS_PAIRING, "rank": rank,
            "time_unix": t, "fetch_ms": fetch_ms, "miss_rows": rows, "rounds": rounds}


# ---- the old solve stays the old solve -------------------------------------

def test_without_the_new_inputs_the_solve_is_the_live_one():
    """bs_weights/miss_ms_rank absent: byte-identical to the live 18:23Z
    publish (the edge model is the launcher's own)."""
    sol = er.solve_owned_cut(solve_at, HAND, 0, miss_ms=POOLED, **KW)
    assert sol.ratios == (215, 121, 152) and sol.cut == (0, 40, 24)
    assert [round(x, 2) for x in sol.round_ms] == [68.78, 13.33, 41.64]
    assert [round(x, 2) for x in sol.base_round_ms] == [62.74, 13.61, 49.89]
    assert (sol.candidates, sol.feasible) == (18700, 2243)
    assert sol.bs_weights == () and sol.round_ms_by_bs[0][0] == 1


# ---- per-card cost from the card's own records -----------------------------

def test_rank_records_give_one_cost_per_card():
    recs = [_rec(0, 202.7, 1000), _rec(1, 473.4, 1000), _rec(2, 217.1, 1000)]
    got = er.owned_miss_per_rank_from_records(recs, n=3)
    assert got is not None
    assert got[0] == pytest.approx(PER_RANK)
    assert "RECORD je Rang" in got[1]
    # the pooled pair hides the x4/x8 gap: one worker cost for both 3080s
    pair, _ = er.owned_miss_from_rank_records(recs, host=0)
    assert pair[1] == pytest.approx((473.4 + 217.1) / 2000.0)


def test_a_missing_rank_keeps_the_pair():
    recs = [_rec(0, 202.7, 1000), _rec(1, 473.4, 1000)]
    assert er.owned_miss_per_rank_from_records(recs, n=3) is None
    thin = [_rec(0, 2.0, 10, rounds=2), _rec(1, 4.0, 10, rounds=2), _rec(2, 2.0, 10, rounds=2)]
    assert er.owned_miss_per_rank_from_records(thin, n=3) is None


def test_round_prices_each_rank_at_its_own_card():
    fits = solve_at(HAND, None)
    a = er.owned_round_ms(fits, host=0, num_experts=NUM_E, ids_per_step=IDS, n_layers=N_LAYERS,
                          miss_ms=(1.0, 1.0))
    b = er.owned_round_ms(fits, host=0, num_experts=NUM_E, ids_per_step=IDS, n_layers=N_LAYERS,
                          miss_ms=(1.0, 1.0), miss_ms_rank=PER_RANK)
    assert b == pytest.approx(tuple(x * c for x, c in zip(a, PER_RANK)))


def test_per_card_cost_moves_the_x4_cards_load_to_the_x8_card():
    """Priced per card the x4 TP1 is twice as dear as the x8 TP2: TP1's
    T_r rises, TP2's falls -- the pooled price had it the other way round."""
    pooled = er.solve_owned_cut(solve_at, HAND, 0, miss_ms=POOLED, **KW)
    per = er.solve_owned_cut(solve_at, HAND, 0, miss_ms=POOLED, miss_ms_rank=PER_RANK, **KW)
    assert per.round_ms[1] > pooled.round_ms[1]
    assert per.round_ms[2] < pooled.round_ms[2]


# ---- the bs mix ------------------------------------------------------------

def test_bs_weights_default_and_override(monkeypatch):
    monkeypatch.delenv("SGLANG_WEG2_OWNED_BS_WEIGHTS", raising=False)
    w, src = er.owned_bs_weights()
    assert w == er.OWNED_BS_WEIGHTS_DEFAULT and "y6k" in src
    assert sum(x for _, x in w) == pytest.approx(1.0, abs=1e-3)
    monkeypatch.setenv("SGLANG_WEG2_OWNED_BS_WEIGHTS", "bs1")
    assert er.owned_bs_weights()[0] is None
    monkeypatch.setenv("SGLANG_WEG2_OWNED_BS_WEIGHTS", "1:1,2:3")
    assert er.owned_bs_weights()[0] == ((1, 0.25), (2, 0.75))
    monkeypatch.setenv("SGLANG_WEG2_OWNED_BS_WEIGHTS", "1:x")
    with pytest.raises(ValueError):
        er.owned_bs_weights()


def test_the_solve_reports_t_r_per_bs_and_holds_x1_at_bs1_and_bs2():
    ob = er.derive_owned_base(solve_at, HAND, 0, miss_ms=POOLED, miss_ms_rank=PER_RANK,
                              bs_weights=er.OWNED_BS_WEIGHTS_DEFAULT, **KW)
    sol = er.solve_owned_cut(solve_at, ob.ratios, 0, miss_ms=POOLED, miss_ms_rank=PER_RANK,
                             bs_weights=er.OWNED_BS_WEIGHTS_DEFAULT, **KW)
    by = dict(sol.round_ms_by_bs)
    base = dict(sol.base_round_ms_by_bs)
    assert sorted(by) == [1, 2, 3]
    for b in (1, 2):
        for w in (1, 2):
            assert by[b][w] <= base[b][w] + 1e-9
    assert sol.objective_ms == pytest.approx(
        sum(w * max(by[b]) for b, w in er.OWNED_BS_WEIGHTS_DEFAULT))
    # bs scales the ids and the verify rows: a bs3 round misses more than bs1
    assert max(by[3]) > max(by[2]) > max(by[1])


# ---- the planner's own base ------------------------------------------------

def test_the_base_does_not_depend_on_the_hand_vector():
    got = {er.derive_owned_base(solve_at, seed, 0, miss_ms=POOLED, miss_ms_rank=PER_RANK,
                                bs_weights=er.OWNED_BS_WEIGHTS_DEFAULT, **KW).ratios
           for seed in (HAND, (163, 163, 162), (215, 121, 152), (240, 100, 148))}
    assert len(got) == 1


def test_the_published_form_beats_the_live_one_at_every_bs():
    """GERECHNET on the live edge: the derived base + per-card cost + bs mix
    publish a form whose max T_r is below the live form's at bs1, bs2 and
    bs3, both priced per card (the live form never ran x1 against itself)."""
    ob = er.derive_owned_base(solve_at, HAND, 0, miss_ms=POOLED, miss_ms_rank=PER_RANK,
                              bs_weights=er.OWNED_BS_WEIGHTS_DEFAULT, **KW)
    new = er.solve_owned_cut(solve_at, ob.ratios, 0, miss_ms=POOLED, miss_ms_rank=PER_RANK,
                             bs_weights=er.OWNED_BS_WEIGHTS_DEFAULT, **KW)
    live = er.solve_owned_cut(solve_at, HAND, 0, miss_ms=POOLED, miss_ms_rank=PER_RANK,
                              bs_weights=er.OWNED_BS_WEIGHTS_DEFAULT, **KW)
    assert live.ratios == (215, 121, 152)
    assert new.ratios != live.ratios
    for (b, ms), (_, lms) in zip(new.round_ms_by_bs, live.round_ms_by_bs):
        assert max(ms) < max(lms), b
    assert new.objective_ms < live.objective_ms


def test_base_mode_env(monkeypatch):
    monkeypatch.delenv("SGLANG_WEG2_OWNED_BASE", raising=False)
    assert er.owned_base_mode() == "derive"
    monkeypatch.setenv("SGLANG_WEG2_OWNED_BASE", "stated")
    assert er.owned_base_mode() == "stated"
    monkeypatch.setenv("SGLANG_WEG2_OWNED_BASE", "hand")
    with pytest.raises(ValueError):
        er.owned_base_mode()


# ---- heat ------------------------------------------------------------------

def _heat_rec(rank, lo, counts, pad=False):
    return {"kind": er.OWNED_HEAT_KIND, "rank": rank,
            "layers": [{"global_lo": lo, "pad": pad, "counts": counts + ([0] if pad else []),
                        "num_local_experts": len(counts) + (1 if pad else 0)}]}


def test_heat_records_map_local_ids_to_global_shares():
    recs = [_heat_rec(0, 0, [3, 1]), _heat_rec(1, 2, [0, 4], pad=True)]
    got = er.owned_heat_from_records(recs, num_experts=4)
    assert got is not None
    assert got[0] == pytest.approx((3 / 8, 1 / 8, 0.0, 4 / 8))
    assert er.owned_heat_from_records([], num_experts=4) is None


def test_hot_resident_experts_miss_less_than_uniform():
    fit = types.SimpleNamespace(rank=0, local_experts=8, ceiling_max_rows=4)
    heat = [0.2] * 4 + [0.0] * 4  # the 4 held rows take every lane
    hot = er.owned_miss_rows_heat(fit, span=(0, 8), heat=heat, ids_per_step=IDS)
    uni = er.owned_miss_rows(fit, num_experts=8, ids_per_step=IDS)
    assert hot == 0.0 and uni > 0.0


# ---- the dry run prints it -------------------------------------------------

def test_plan_d_residency_prints_the_derived_base_and_t_r_per_bs():
    src = inspect.getsource(er.plan_d_residency)
    assert "D-EIGENTUM BASIS (01.10., Planer statt Hand)" in src
    assert "D-EIGENTUM T_r JE BS (01.10.)" in src
    assert "derive_owned_base(" in src and "owned_base_mode()" in src
    for key in ('"base_mode"', '"stated_ratios"', '"miss_ms_per_rank"', '"round_ms_by_bs"',
                '"routing"'):
        assert key in src


def test_the_launcher_hands_the_per_card_cost_and_heat_to_every_pass():
    from sglang.srt.weg2 import launcher

    src = inspect.getsource(launcher.log_d_rank_vram_solve)
    assert src.count("owned_miss_ms_rank=_miss_rank") == src.count("owned_miss_ms=_miss_ms") == 3
    assert src.count("owned_heat_records=_heat_recs") == 3


# ---- the whole planner pass (the dry run's lines) --------------------------

MODEL = "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"


@pytest.fixture
def ckpt(tmp_path, monkeypatch):
    import json

    from sglang.srt.planner import pp_cut

    cfg = {"text_config": {"num_hidden_layers": 48, "vocab_size": 248320,
                           "hidden_size": 2560}}
    (tmp_path / MODEL).mkdir(parents=True)
    (tmp_path / MODEL / "config.json").write_text(json.dumps(cfg))
    monkeypatch.setattr(
        pp_cut, "checkpoint_weight_terms",
        lambda _p: types.SimpleNamespace(expert_layer_weight_bytes=1297637376.0,
                                         num_experts=512, n_layers=48))
    return str(tmp_path / MODEL)


def _owned_plan(ckpt, **kw):
    return er.plan_d_residency(
        model_path=ckpt, budgets_mib=[26824, 17640, 17520], ratios=[183.0, 137.0, 168.0],
        fractions=[0.06, 0.51, 0.48], scratch_rows=[104, 48, 48], rank_tp_ratio="1,0,0",
        env_d={"SGLANG_UNEVEN_MOE_EXPERT_SHARD": "1",
               "SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL": "1"},
        reference_logs="", kv_tokens=262144, label="T", marker="T",
        kv_token_shares=er.KV_TOKEN_CUT_OWNED, kv_dcp_cell_bytes=12288,
        owned_miss_ms=POOLED, owned_miss_source="test", **kw)


def test_the_planner_pass_derives_the_base_and_prints_t_r_per_bs(ckpt, monkeypatch):
    monkeypatch.delenv("SGLANG_WEG2_OWNED_BASE", raising=False)
    monkeypatch.delenv("SGLANG_WEG2_OWNED_BS_WEIGHTS", raising=False)
    plan = _owned_plan(ckpt, owned_miss_ms_rank=PER_RANK, owned_miss_rank_source="test")
    base = [ln for ln in plan.lines if "D-EIGENTUM BASIS (01.10., Planer statt Hand)" in ln]
    # the toy checkpoint carries no cut form (no layer_types: 48 FA layers);
    # the derived base and its T_r per bs are printed all the same
    assert base
    assert "Saat 183,137,168, nur Startpunkt" in base[0]
    assert "je Rang ['0.203', '0.473', '0.217']" in base[0]
    assert "gleichverteilt (kein Hitze-Record" in base[0]
    assert "T_r je bs bs1 " in base[0] and " bs2 " in base[0] and " bs3 " in base[0]
    rec = dict(plan.owner_record)
    assert rec["base_mode"] == "derive" and rec["stated_ratios"] == [183, 137, 168]
    assert [b for b, _ in rec["base_round_ms_by_bs"]] == [1, 2, 3]


def test_stated_and_bs1_are_the_explicit_override(ckpt, monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_OWNED_BASE", "stated")
    monkeypatch.setenv("SGLANG_WEG2_OWNED_BS_WEIGHTS", "bs1")
    plan = _owned_plan(ckpt)
    assert not any("D-EIGENTUM BASIS" in ln or "T_r JE BS" in ln for ln in plan.lines)
    rec = dict(plan.owner_record)
    assert rec["base_mode"] == "stated" and rec["bs_weights"] is None
