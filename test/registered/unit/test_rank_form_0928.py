"""The rank form (28.09.): one mechanism, forms A/B/C are values of two
per-rank vectors; the planner picks from measured prices only."""

import json

import pytest

from sglang.srt import rank_form as rf

U = ["GPU-5090", "GPU-3080x8", "GPU-3080x4"]


def test_forms_are_values_of_the_vectors():
    assert rf.resolve_rank_form([1, 1, 1]).kind == rf.FORM_C
    assert rf.resolve_rank_form([98, 19, 19]).kind == rf.FORM_C
    a = rf.resolve_rank_form([1, 0, 0], [0, 50, 50])
    assert (a.kind, a.head_rank, a.kv_only_ranks) == (rf.FORM_A, 0, (1, 2))
    b = rf.resolve_rank_form([77, 23, 0], [10, 45, 45], subgroup_tp_wired=True)
    assert (b.kind, b.weight_ranks, b.kv_only_ranks) == (rf.FORM_B, (0, 1), (2,))


def test_form_a_backend_follows_the_model():
    # the same #239 vectors: dense GQA -> the weightless lane, MoE/QSA -> #239 workers
    dense = rf.resolve_rank_form([1, 0, 0], [0, 50, 50], rank_role="host,worker,worker")
    assert dense.backend == rf.BACKEND_LANE
    moe = rf.resolve_rank_form([1, 0, 0], [0, 50, 50], rank_role="host,worker,worker", dense=False)
    assert moe.backend == rf.BACKEND_FORM_A_239
    assert moe.serve_argv() == ["--tp-size", "3", "--rank-role", "host,worker,worker",
                                "--rank-tp-ratio", "1,0,0", "--uneven-token-vector", "0,50,50"]
    # #239's own role check decides disagreement
    with pytest.raises(rf.RankFormShapeMismatch):
        rf.resolve_rank_form([1, 1, 0], None, rank_role="host,worker,worker")
    rf.check_form_spec(moe, "DFLASH")  # not the lane: the lane's spec rule does not apply


def test_form_a_runs_on_the_existing_lane():
    a = rf.resolve_rank_form([0, 1, 0])
    assert a.serve_argv() == ["--tp-size", "3", "--dcp-size", "3", "--weightless-kv-fastlane",
                              "--weightless-kv-head-rank", "1"]
    c = rf.resolve_rank_form([98, 19, 19], [40, 30, 30])
    assert c.serve_argv() == ["--tp-size", "3", "--rank-tp-ratio", "98,19,19",
                              "--uneven-token-vector", "40,30,30"]
    assert rf.resolve_rank_form([1, 1]).serve_argv() == ["--tp-size", "2"]


@pytest.mark.parametrize("w,t,cls", [
    ([0, 0, 0], None, rf.RankFormNoWeightRank),
    ([1, 0], [1, 1, 1], rf.RankFormShapeMismatch),
    ([1, 0, 0], [50, 50, 0], rf.RankFormShapeMismatch),   # KV-only rank holding nothing
    ([1, -1, 0], None, rf.RankFormShapeMismatch),
    ([77, 23, 0], None, rf.RankFormSubgroupTpNotWired),   # B needs F6
])
def test_nonsense_is_refused_by_name(w, t, cls):
    with pytest.raises(cls) as ei:
        rf.resolve_rank_form(w, t)
    assert str(ei.value).startswith(cls.code)


def test_vision_needs_a_tail_on_the_head():
    a = rf.resolve_rank_form([1, 0, 0], [0, 50, 50])
    with pytest.raises(rf.RankFormVisionNoTail):
        rf.check_form_vision(a, vision=True)
    assert "own lease 512 MiB" in rf.check_form_vision(a, vision=True, head_lease_mib=512)
    assert rf.check_form_vision(rf.resolve_rank_form([1, 0, 0], [10, 45, 45]), vision=True) is None
    assert rf.check_form_vision(a, vision=False) is None


def test_lane_spec_is_the_eagle_chain_only():
    a = rf.resolve_rank_form([1, 0, 0])
    rf.check_form_spec(a, "NEXTN")
    rf.check_form_spec(a, None)
    with pytest.raises(rf.RankFormLaneSpec):
        rf.check_form_spec(a, "DFLASH")
    rf.check_form_spec(rf.resolve_rank_form([1, 1, 1]), "DFLASH")  # C: not the lane's business


def _measures(tmp_path, ar=None, dcp=None):
    doc = {"schema": rf.MEASURES_SCHEMA, "allreduce": ar or {}, "dcp_exchange": dcp or {}}
    p = tmp_path / "m.json"
    p.write_text(json.dumps(doc))
    return rf.FormMeasures.load(str(p))


def _cost(form, m, bs=1):
    return rf.form_round_cost(form, bs=bs, weight_read_ms=[10.0, 25.0, 25.0],
                              attn_ms=[1.0, 4.0, 4.0], fixed_ms=2.0, n_allreduce=98,
                              n_attn_layers=16, uuids=U, measures=m)


def test_unmeasured_keeps_todays_form(tmp_path):
    m = rf.FormMeasures.load(None)
    cands = {"A": rf.resolve_rank_form([1, 0, 0], [0, 50, 50]),
             "C": rf.resolve_rank_form([98, 19, 19])}
    costs = {k: _cost(f, m) for k, f in cands.items()}
    assert costs["A"].round_ms is None and costs["A"].missing
    label, line = rf.choose_form(cands, costs, today="C")
    assert label == "C" and "UNMEASURED" in line and "dcp_exchange[" in line


def test_measured_prices_decide(tmp_path):
    key3 = rf.cards_key(U)
    m = _measures(tmp_path,
                  ar={key3: {"1": {"graph_median_us": 100.0}}},
                  dcp={key3: {"1": {"graph_median_us": 50.0, "eager_median_us": 90.0}}})
    a = rf.resolve_rank_form([1, 0, 0], [0, 50, 50])
    c = rf.resolve_rank_form([98, 19, 19])
    ca, cc = _cost(a, m), _cost(c, m)
    # A: max(5090 reads all weights 10 ms, each 3080 attends half the KV 4*0.5)
    #    + fixed 2 + 16 * 0.05 ms dcp, no allreduce
    assert ca.round_ms == pytest.approx(10.0 + 2.0 + 0.8)
    assert "allreduce_ms" not in ca.terms
    # C: max rank compute + 98 * 0.1 ms allreduce + dcp
    assert cc.terms["allreduce_ms"] == pytest.approx(9.8)
    label, line = rf.choose_form({"A": a, "C": c}, {"A": ca, "C": cc}, today="C")
    assert label == "A" and "CHOSEN A" in line


def test_override_wins_by_name(tmp_path):
    m = rf.FormMeasures.load(None)
    cands = {"A": rf.resolve_rank_form([1, 0, 0]), "C": rf.resolve_rank_form([1, 1, 1])}
    costs = {k: _cost(f, m) for k, f in cands.items()}
    label, line = rf.choose_form(cands, costs, today="C", override="A")
    assert label == "A" and "FORM-OVERRIDE A" in line
    with pytest.raises(ValueError):
        rf.choose_form(cands, costs, today="C", override="B")


def test_measures_schema_is_checked(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"schema": "other/1"}))
    with pytest.raises(ValueError):
        rf.FormMeasures.load(str(p))
