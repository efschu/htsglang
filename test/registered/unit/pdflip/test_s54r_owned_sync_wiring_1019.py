"""S54r (review 19.10.): the S54 owned solve through the WHOLE planner pass.

The S54 unit tests prove ``solve_owned_cut``; three things only the planner
pass (``plan_d_residency``) and the launcher do were held by nothing but the
box-bound NF golden (review mutants M6 / M14 survived the S54 tests):

* the launcher hands each D rank's card-probe link to every solve pass;
* without a rank record the pair cost is split over the cards by that link
  (a rank record still wins);
* ``x1=sync`` without ANY per-card price (no rank record, no link) would run
  on one pooled worker price for both 3080s -- the 29.09. ``round`` metal
  loss (misses and KV cut on the x4 card). On the toy edge below sync then
  publishes the whole KV cut on TP1 (cut [0, 64, 0]); the pass now holds x1
  per worker for that solve and names it (danger direction of the review).
"""

from __future__ import annotations

import inspect
import json
import types

import pytest

from flliper.srt.planner import expert_residency as er
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-test-cpu")

MODEL = "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"
POOLED = (0.201074, 0.314024)
PER_RANK = (0.2027, 0.4734, 0.2171)
#: card probe card_probe-20ae5edfc9b2.json, pinned H2D GB/s, D rank order (5090, 3080 x4, 3080 x8)
H2D = [14.42, 6.47, 13.40]


@pytest.fixture
def ckpt(tmp_path, monkeypatch):
    from flliper.srt.planner import pp_cut

    cfg = {"text_config": {"num_hidden_layers": 48, "vocab_size": 248320,
                           "hidden_size": 2560, "num_experts_per_tok": 10}}
    (tmp_path / MODEL).mkdir(parents=True)
    (tmp_path / MODEL / "config.json").write_text(json.dumps(cfg))
    monkeypatch.setattr(
        pp_cut, "checkpoint_weight_terms",
        lambda _p: types.SimpleNamespace(expert_layer_weight_bytes=1297637376.0,
                                         num_experts=512, n_layers=48))
    for k in ("FLLIPER_PDFLIP_OWNED_BS_WEIGHTS", er.OWNED_X1_ENV,
              "FLLIPER_PDFLIP_OWNED_EXTEND_TOKENS", "FLLIPER_PDFLIP_OWNED_EXTEND_PER_ROUND"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("FLLIPER_PDFLIP_OWNED_BASE", "stated")
    return str(tmp_path / MODEL)


def _plan(ckpt, **kw):
    plan = er.plan_d_residency(
        model_path=ckpt, budgets_mib=[26824, 17640, 17520], ratios=[183.0, 137.0, 168.0],
        fractions=[0.06, 0.51, 0.48], scratch_rows=[104, 48, 48], rank_tp_ratio="1,0,0",
        env_d={"FLLIPER_UNEVEN_MOE_EXPERT_SHARD": "1",
               "FLLIPER_PDFLIP_DENSE_REPACK_OUTSIDE_POOL": "1"},
        reference_logs="", kv_tokens=262144, label="T", marker="T",
        kv_token_shares=er.KV_TOKEN_CUT_OWNED, kv_dcp_cell_bytes=12288,
        owned_miss_ms=POOLED, owned_miss_source="test", **kw)
    return plan, dict(plan.owner_record or ())


def test_the_launcher_hands_the_link_to_every_pass():
    from flliper.srt.pdflip import launcher

    src = inspect.getsource(launcher.log_d_rank_vram_solve)
    assert src.count("owned_extend_h2d_gbps=_ext_gbps") == src.count("owned_miss_ms=_miss_ms") == 3


def test_the_pass_prices_the_extend_and_splits_the_pair_by_the_link(ckpt):
    plan, rec = _plan(ckpt, owned_extend_h2d_gbps=H2D, owned_extend_source="probe")
    row = er.owned_extend_row_ms(H2D, 2534448)
    assert rec["x1_scope"] == "sync" and not rec["x1_pooled_fallback"]
    assert rec["extend_row_ms"] == [round(x, 4) for x in row]
    assert rec["extend_ms"] and rec["extend_per_round"] == er.OWNED_EXTEND_PER_ROUND_SEED
    assert rec["miss_ms_per_rank"] == pytest.approx(er.owned_miss_rank_from_link(POOLED, row, 0))
    assert "Kartenprobe-Link" in rec["miss_ms_rank_source"]
    ext = [ln for ln in plan.lines if "D-EIGENTUM EXTEND-KOPIE (S54)" in ln]
    assert len(ext) == 1 and "aus Paar x Link abgeleitet" in ext[0]


def test_a_rank_record_wins_over_the_link(ckpt):
    _, rec = _plan(ckpt, owned_extend_h2d_gbps=H2D, owned_miss_ms_rank=PER_RANK,
                   owned_miss_rank_source="rec")
    assert rec["miss_ms_per_rank"] == list(PER_RANK) and rec["miss_ms_rank_source"] == "rec"
    assert rec["x1_scope"] == "sync" and rec["extend_ms"]


def test_sync_without_any_per_card_price_holds_x1_per_worker(ckpt, monkeypatch):
    plan, rec = _plan(ckpt)
    assert rec["x1_scope"] == "workers" and rec["x1_pooled_fallback"]
    assert rec["miss_ms_per_rank"] is None and rec["extend_ms"] is None
    assert any("D-EIGENTUM X1 (S54r)" in ln and "workers" in ln for ln in plan.lines)
    assert any("EXTEND-KOPIE (S54): AUS" in ln for ln in plan.lines)
    # the same form as the explicit workers scope -- never sync on a pooled price
    monkeypatch.setenv(er.OWNED_X1_ENV, "workers")
    _, w = _plan(ckpt)
    assert (rec["ratios"], rec["cut"]) == (w["ratios"], w["cut"])
    assert not w["x1_pooled_fallback"]
    # the danger it holds off: sync on the pooled price hands the x4 TP1 the whole cut
    assert tuple(rec["cut"]) != (0, 64, 0)
