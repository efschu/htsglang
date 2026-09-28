"""#251c launcher and capture: D's KV stage form as form values, and the
captured decode steps that stay valid in every phase the form allows.

WHAT MUST HOLD.
(1) The table: stage tokens S0 x (1, 1.5, 2); a stage row is an expert row
    minus its scales; the stage rows ON in the boot form are the top stage's
    + 1; per seat count n the highest stage whose rows keep every batch b <= n
    at the waves it has WITHOUT stages (variant B, the default until the wave
    measurement prices an extra wave).
(2) The launcher writes the form into --env-d: TP0 moves the stage rows from
    its scratch into its seat rows (the boot maps the same bank); a second
    solve pass takes its own write back first (idempotent); an operator's
    stage tokens win.
(3) The capture floor per bs = the fewest rows ON of any phase that batch can
    replay in; with it every captured batch keeps its stage-free wave count.
(4) A captured step counts the floor's rows as ON (the bound and the wave
    count); outside a capture the live count, byte-identical.
"""
from __future__ import annotations

import os
import types
from unittest import mock

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from sglang.srt.layers.moe import expert_pool_device as ep  # noqa: E402
from sglang.srt.planner import expert_residency as er  # noqa: E402
from sglang.srt.weg2 import d_seat_vram as dsv  # noqa: E402

NF_MODEL = (
    "/spinning/llm_stuff/club-3090/models-cache/"
    "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"
)
FORM = er.SeatVramForm(temporal_slot_bytes=(48 * 128 * 128 * 2, 0, 0), gdn_layers=36,
                       expert_row_bytes=2534448, moe_layers=48, small_row_bytes=76848)


def _rows():
    """The H95c fixture: scratch 100 on TP0, R = 120 - 100 = 20 at the cap,
    seat extras 13/10/8/5/1/0."""

    def row(n, max_rows):
        return er.SeatTableRow(
            seats=n, ids_per_step=40 * n, waves=2, mamba_slots=7, host_mamba_mib=0.0,
            host_spec_mib=0.0, max_rows=max_rows, scratch_given=(100, 48, 48),
            waves_given=(1, 1, 1), fraction_given=(None, None, None),
            scratch_min=(None, None, None), fraction_max=(None, None, None), refusal=None)

    return er._seat_vram_columns(
        tuple(row(n, (136 - (16 * (n - 1)) // 5, 140, 141)) for n in range(1, 7)), FORM)


def _table():
    return er.kv_stage_table(_rows(), FORM, kv_cell_bytes=14143, kv_tokens=262144,
                             local_experts=193, verify_tokens=4, top_k=10, staging_rows=12)


# ---- (1) the table ------------------------------------------------------------------

def test_the_stage_table_keeps_every_batch_at_its_waves():
    t = _table()
    assert t.tokens == (262144, 393216, 524288)
    assert t.row_mib == 112.5  # (2534448 - 76848) B x 48 layers
    assert t.stage_rows == (0, 16, 32) and t.rows == 33
    # demand D(b) = min(40 b, 173), today's waves over C = 100: 1,1,2,2,2,2
    assert t.need == (40, 80, 80, 80, 87, 87)
    assert [c[0] for c in t.capacity] == [113, 110, 108, 105, 101, 100]
    assert t.max_by_seats == (2, 1, 1, 1, 0, 0)
    # the default adds no wave to any batch
    assert t.waves == (1, 1, 2, 2, 2, 2) and t.extra_waves(t.max_by_seats) == (0,) * 6
    # the A/B candidate S1 up to the cap: the 6-seat phase at S1 (84 rows)
    # replays the bs2 graph too -- bs2 (80 ids) keeps 1 wave only at 84 >= 80,
    # bs4 (160) needs 2, bs5/6 (173) need 3: one extra wave at bs5/6 only
    assert t.extra_waves((2, 1, 1, 1, 1, 1)) == (0, 0, 0, 0, 1, 1)
    # variant A (the top stage everywhere) adds waves from bs2 on
    assert t.extra_waves((2,) * 6) == (0, 1, 0, 1, 1, 1)
    assert er.kv_stage_table(_rows(), FORM, kv_cell_bytes=0, kv_tokens=262144,
                             local_experts=193, verify_tokens=4, top_k=10) is None
    # the scratch cannot give the stage rows and keep its staging: no form
    assert er.kv_stage_table(_rows(), FORM, kv_cell_bytes=14143, kv_tokens=262144,
                             local_experts=193, verify_tokens=4, top_k=10,
                             staging_rows=70) is None


# ---- (2) the launcher -------------------------------------------------------------

def _plan():
    return types.SimpleNamespace(fits=[types.SimpleNamespace(
        rank=0, kv_cell_bytes=14143, kv_tokens=262144, local_experts=193, staging_rows=12)])


def test_the_launcher_writes_the_form_and_takes_it_back():
    from sglang.srt.weg2 import launcher as L

    before = "SGLANG_MOE_SCRATCH_SLOTS=100,48,48;SGLANG_WEG2_D_SEAT_EXPERT_ROWS=14,0,0;X=1"
    ns = types.SimpleNamespace(env_d=before)
    lines = L.apply_d_kv_stage_form(ns, er, _rows(), FORM, _plan(), "D", verify_tokens=4,
                                    top_k=10)
    env = L.parse_group_env(ns.env_d)
    assert env["SGLANG_MOE_SCRATCH_SLOTS"] == "67,48,48"
    assert env["SGLANG_WEG2_D_SEAT_EXPERT_ROWS"] == "47,0,0"
    assert env["SGLANG_WEG2_D_KV_STAGE_TOKENS"] == "262144,393216,524288"
    assert env["SGLANG_WEG2_D_KV_STAGE_ROWS"] == "33"
    assert env["SGLANG_WEG2_D_KV_STAGE_MAX_BY_SEATS"] == "2,1,1,1,0,0"
    assert env["X"] == "1"
    assert "D-KV-STUFEN (#251c)" in lines[0] and "Scratch 100 -> 67" in lines[0]
    assert "Variante B" in lines[1]
    assert "MAX_BY_SEATS=2,1,1,1,1,1" in lines[2] and "[0, 0, 0, 0, 1, 1]" in lines[2]
    # a second solve pass: its own write goes back first, then it writes again
    L.d_kv_stage_undo(ns)
    assert L.parse_group_env(ns.env_d) == L.parse_group_env(before)
    L.apply_d_kv_stage_form(ns, er, _rows(), FORM, _plan(), "D", verify_tokens=4, top_k=10)
    assert L.parse_group_env(ns.env_d)["SGLANG_MOE_SCRATCH_SLOTS"] == "67,48,48"
    # no seat rows stated before: the undo removes the key again
    ns2 = types.SimpleNamespace(env_d="SGLANG_MOE_SCRATCH_SLOTS=100,48,48")
    L.apply_d_kv_stage_form(ns2, er, _rows(), FORM, _plan(), "D", verify_tokens=4, top_k=10)
    assert L.parse_group_env(ns2.env_d)["SGLANG_WEG2_D_SEAT_EXPERT_ROWS"] == "33,0,0"
    L.d_kv_stage_undo(ns2)
    assert ns2.env_d == "SGLANG_MOE_SCRATCH_SLOTS=100,48,48"


def test_an_operators_stage_tokens_win():
    from sglang.srt.weg2 import launcher as L

    told = "SGLANG_MOE_SCRATCH_SLOTS=100,48,48;SGLANG_WEG2_D_KV_STAGE_TOKENS=262144"
    ns = types.SimpleNamespace(env_d=told)
    (line,) = L.apply_d_kv_stage_form(ns, er, _rows(), FORM, _plan(), "D", verify_tokens=4,
                                      top_k=10)
    assert "selbst" in line and ns.env_d == told
    L.d_kv_stage_undo(ns)  # nothing of its own to take back
    assert ns.env_d == told


def test_the_solve_takes_back_its_own_form_before_it_prices():
    from sglang.srt.weg2 import launcher as L

    src = open(L.__file__).read()
    i = src.index("def log_d_rank_vram_solve(")
    j = src.index("d_kv_stage_undo(ns)", i)
    assert j < src.index("parse_group_env(getattr(ns, \"env_d\"", i)
    k = src.index("def d_seat_table_lines(")
    body = src[k:k + 3000]
    assert body.index("apply_d_seat_expert_rows(") < body.index("apply_d_kv_stage_form(")


@pytest.mark.skipif(not os.path.isdir(NF_MODEL), reason="needs the NF checkpoint header")
def test_dry_run_of_the_nf_form_writes_the_stage_form():
    from sglang.srt.weg2 import launcher as L

    ns = types.SimpleNamespace(model=NF_MODEL, profile=L.PROFILE_NEXTFLASH, d_bs=6,
                               extra_d="--max-running-requests 6",
                               env_d="SGLANG_MOE_SCRATCH_SLOTS=100,48,48")
    L.apply_profile_d_seat_vram_default(ns)
    L.apply_profile_d_pool_waves_default(ns)
    env_d = dict(L.parse_group_env(ns.env_d), SGLANG_UNEVEN_MOE_EXPERT_SHARD="1",
                 SGLANG_MOE_OFFLOAD_GRAPH_MODE="pool", SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL="1",
                 SGLANG_WEG2_DRAFT_SHARE_EMBED="1")
    kw = dict(model_path=NF_MODEL, budgets_mib=[29624, 18664, 18672], ratios=[183, 137, 168],
              fractions=[0.06, 0.51, 0.48], scratch_rows=[100, 48, 48], rank_tp_ratio="1,0,0",
              env_d=env_d, reference_logs="", kv_tokens=262144, label="D",
              marker=L.D_RANK_SOLVE_MARKER, card_reference_logs="", seat_graph_mib=None,
              reference_seats=1)
    with mock.patch.object(L, "d_replayssm_spec_plan_form", lambda _ns: er.ReplaySSMSpecForm(
            ring_len=16, draft_tokens=4, max_running=6, ssm_dtype="bfloat16")):
        lines = L.d_seat_table_lines(ns, er, kw, "D")
    env = L.parse_group_env(ns.env_d)
    rows = int(env["SGLANG_WEG2_D_KV_STAGE_ROWS"])
    assert env["SGLANG_WEG2_D_KV_STAGE_TOKENS"] == "262144,393216,524288"
    # #239 S3g floor (rc12z30b 28.09. 20:09:07): a byteless Form A worker holds no
    # KV (no QSA keys, qsa_index_on_rank=False), so it gets no stage rows
    assert env["SGLANG_WEG2_D_SEAT_EXPERT_ROWS"] == "%d,0,0" % (14 + rows)
    assert env["SGLANG_MOE_SCRATCH_SLOTS"] == "%d,48,48" % (100 - rows)
    assert "SGLANG_WEG2_D_KV_STAGE_ROWS_BY_RANK" not in env
    assert len(env["SGLANG_WEG2_D_KV_STAGE_MAX_BY_SEATS"].split(",")) == 6
    assert any("D-KV-STUFEN (#251c)" in ln for ln in lines)


# ---- (3) the capture floor ----------------------------------------------------

def _cells(extra=(13, 10, 8, 5, 1, 0), stage_rows=(0, 16, 32), S=33):
    return {(n, j): types.SimpleNamespace(extra_rows=extra[n - 1] + S - r)
            for n in range(1, 7) for j, r in enumerate(stage_rows)}


def test_the_capture_floor_keeps_every_batch_at_its_waves():
    form = dsv.StageForm(tokens=(262144, 393216, 524288), rows_on=33,
                         max_by_seats=(2, 1, 1, 1, 0, 0))
    floors = dsv.capture_floors(form, _cells(), 6)
    assert floors == (14, 22, 22, 22, 33, 33)
    # with those rows ON the captured step of b has today's waves (C = 100)
    c_min = 100 - 33
    for b, f in enumerate(floors, start=1):
        d = min(40 * b, 173)
        assert ep.pool_waves_for(40 * b, 193, 20, c_min + f) == ep.pool_waves_for(
            40 * b, 193, 20, 100), b
        assert d <= ep.pool_waves_for(40 * b, 193, 20, c_min + f) * (c_min + f)
    # variant A (every stage at every n) would floor at the top stage: extra waves
    all_top = dsv.capture_floors(dsv.StageForm(tokens=form.tokens, rows_on=33), _cells(), 6)
    assert all_top == (1, 1, 1, 1, 1, 1)
    assert ep.pool_waves_for(240, 193, 20, c_min + 1) > ep.pool_waves_for(240, 193, 20, 100)


def test_capture_floor_rows_maps_ids_to_the_batch():
    with mock.patch.dict(dsv._CAPTURE, {"runner": None, "floors": (14, 22, 22, 22, 33, 33)}), \
            mock.patch.object(dsv, "stage_form", lambda env=None: object()):
        assert dsv.capture_floor_rows(40, 40) == 14
        assert dsv.capture_floor_rows(80, 40) == 22
        assert dsv.capture_floor_rows(240, 40) == 33
        # a draft step (10 ids per seat) reads as a SMALLER batch: a lower floor
        assert dsv.capture_floor_rows(60, 40) == 22 and dsv.capture_floor_rows(20, 40) == 14
        # past the seat cap (an extend shape) and without per-seat ids: live count
        assert dsv.capture_floor_rows(4000, 40) == 0
        assert dsv.capture_floor_rows(40, None) == 0
    with mock.patch.dict(dsv._CAPTURE, {"runner": None, "floors": (14,)}):
        assert dsv.capture_floor_rows(40, 40) == 0  # no form: nothing


# ---- (4) the captured bound --------------------------------------------------------

def test_a_captured_step_counts_the_floor_rows_as_on():
    E, R, C, S, X = 60, 3, 7, 2, 40
    hot = {e: e for e in range(R)}
    t = ep.allocate_pool_tables("cpu", E, R + C + X, R, S, hot,
                                [(-1 if e in hot else e) for e in range(E)], seat_rows=X)
    live = ep.pool_row_capacity(t)
    assert ep.pool_row_capacity(t, 0) == live
    assert ep.pool_row_capacity(t, 10) == live + 10
    assert ep.pool_row_capacity(t, 99) == live + X  # never past the seat rows
    buf = ep.allocate_step_buffers("cpu", E, 64)
    ids = torch.full((live + 5,), 5, dtype=torch.int32)
    with pytest.raises(ValueError, match="Step ids exceed the LRU rows"):
        ep.step(t, ids, buf)
    ep.step(t, ids, buf, capture_rows_on=10)
    # with the rows really ON the floor adds nothing
    ep.set_seat_rows_on(t, 20, device_write=False)
    assert ep.pool_row_capacity(t, 10) == ep.pool_row_capacity(t) == live + 20


# ---- (4) the refusal names its reason (YaRN x2 dry run 28.09.) ---------------------

def test_no_stage_form_names_why():
    """S0 = 524288 (YaRN x2): the top stage needs 64 rows; with 40 staging rows
    the scratch of 100 cannot give them -- the reason is C - S <= staging, not
    a missing KV cell."""
    why = []
    assert er.kv_stage_table(_rows(), FORM, kv_cell_bytes=14143, kv_tokens=524288,
                             local_experts=193, verify_tokens=4, top_k=10,
                             staging_rows=40, why=why) is None
    (reason,) = why
    assert "C 100 - S 64 = 36 <= Staging 40" in reason
    assert "524288,786432,1048576" in reason and "D faehrt fest 524288 Token" in reason
    why = []
    assert er.kv_stage_table(_rows(), FORM, kv_cell_bytes=0, kv_tokens=524288,
                             local_experts=193, verify_tokens=4, top_k=10, why=why) is None
    assert why and "keine KV-Zelle" in why[0]


def test_the_launcher_line_says_the_scratch_is_too_small():
    from sglang.srt.weg2 import launcher as L

    plan = types.SimpleNamespace(fits=[types.SimpleNamespace(
        rank=0, kv_cell_bytes=14143, kv_tokens=524288, local_experts=193, staging_rows=40)])
    ns = types.SimpleNamespace(env_d="SGLANG_MOE_SCRATCH_SLOTS=100,48,48")
    (line,) = L.apply_d_kv_stage_form(ns, er, _rows(), FORM, plan, "D", verify_tokens=4,
                                      top_k=10)
    assert "D-KV-STUFEN (#251c): entfaellt -- Scratch zu klein" in line
    assert "keine KV-Zelle" not in line
    assert ns.env_d == "SGLANG_MOE_SCRATCH_SLOTS=100,48,48"
