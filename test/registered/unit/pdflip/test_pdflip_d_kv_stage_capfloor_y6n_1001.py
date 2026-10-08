"""y6n (01.10. 21:07:02, NF, image b0bf738b39, OWNED_BASE=stated): D TP2 died
twice in the decode capture, 'Step ids exceed the LRU rows plus the staging rows'.

The rank (D.log ~5157-5160):
  CAPTURE-FLOOR rows ON per bs [0, 0, 0, 0, 0, 0]
  residents 67, LRU rows 38, staging 8, spill rows 93, seat rows 36
  captured step of 240 ids -> 2 wave(s); min(ids, E-R)=93, C=LRU+staging=46
-> 93 > 2 x 46. The launcher (front.log 244) had passed it:
  WELLENBODEN rang2 D 93 <= 2 x (38 + 9) = 94, Luft 1

WHY (a): ``_stage_capture_rows`` = min(seat rows 36 - lend head 16, floor) --
the lend head is not it (20 rows remain); the floor is 0 because
``capture_floors`` takes the fewest rows ON over every (n >= b, stage) cell
(by demand: every stage at every n), and TP2's top cell (524288 tokens) funds
k = 0: its KV grows by 491520 x 4608 B = 19.2 stage rows of 112.5 MiB over the
BORN floor stage (32768), all 20 of its stage rows go.

WHY (b): ``kv_stage_table`` counted the stage rows from the floor in two
rounded pieces: low = floor(229376 x 4608 / row) = 8 (8.96) plus
ceil(262144 x 4608 / row) = 11 (10.24) = 19 for the top, so its capacity at
the top was 58 + 8 - 19 = 47 (one row ON) -- one row more than the rank's cell
(ceil(19.2) = 20). The launcher's floor read that 47 (printed 38 + 9: the
scratch without the low rows, + staging and low folded into 'on'), 2 x 47 >= 93.

FIX: one ceil over the whole span from the born stage (the rank's own
arithmetic), the spare row on top of it (rows = top + 1) -- the table's
capacity at the top is the rank's C, and the launcher raises the wave cap
(derived) or refuses the form by name (W169) instead of the capture dying.
"""

from __future__ import annotations

import os
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.layers.moe import expert_pool_device as ep  # noqa: E402
from flliper.srt.planner import expert_residency as er  # noqa: E402
from flliper.srt.pdflip import d_seat_vram as dsv  # noqa: E402
from flliper.srt.pdflip import launcher as L  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

#: the NF stage row: expert row minus its scales x 48 MoE layers = 112.5 MiB
FORM = er.SeatVramForm(temporal_slot_bytes=(48 * 128 * 128 * 2, 0, 0), gdn_layers=36,
                       expert_row_bytes=2534448, moe_layers=48, small_row_bytes=76848)
ROW = (2534448 - 76848) * 48
FLOOR = 32768
S0 = 262144
TOP = 524288
#: y6n second solve pass (front.log 236-249): rows per rank per seat count,
#: scratch given, TP0's seat extras; the token cut 0,40,24 of 64
MAX_ROWS = ((149, 117, 125), (146, 117, 125), (143, 117, 125), (140, 117, 125),
            (136, 117, 125), (133, 117, 125))
SCRATCH = (107, 65, 58)
EXTRA = (12, 10, 7, 5, 1, 0)
E = (227, 128, 160)
CELLS = (1855, 7680, 4608)  # host cell, worker trim cells (share x 12288)
ENV_D = ("FLLIPER_MOE_SCRATCH_SLOTS=107,65,58;FLLIPER_PDFLIP_D_SEAT_EXPERT_ROWS=13,0,0;"
         "FLLIPER_PDFLIP_D_KV_STAGE_FLOOR_TOKENS=32768;FLLIPER_PDFLIP_D_KV_STAGE_BY_DEMAND=1;"
         "FLLIPER_OPT_MOE_POOL_OVERFLOW_WAVES=2")
#: TP2 at the rank: residents, ids of the captured step at six seats, lend head
R2, IDS, LEND_HEAD = 67, 240, 16


def _rows():
    def row(n):
        return er.SeatTableRow(
            seats=n, ids_per_step=40 * n, waves=2, mamba_slots=7, host_mamba_mib=0.0,
            host_spec_mib=0.0, max_rows=MAX_ROWS[n - 1], scratch_given=SCRATCH,
            waves_given=(2, 2, 2), fraction_given=(None, None, None),
            scratch_min=(None, None, None), fraction_max=(None, None, None), refusal=None,
            seat_extra=(EXTRA[n - 1], 0, 0))

    return tuple(row(n) for n in range(1, 7))


def _plan():
    return types.SimpleNamespace(fits=[
        types.SimpleNamespace(rank=r, kv_cell_bytes=CELLS[0] if r == 0 else 0,
                              kv_tokens=S0, local_experts=E[r], staging_rows=12,
                              kv_stage_cell_bytes=0 if r == 0 else CELLS[r])
        for r in range(3)])


def _tp2_table():
    return er.kv_stage_table(_rows(), FORM, kv_cell_bytes=CELLS[2], kv_tokens=S0,
                             local_experts=E[2], verify_tokens=4, top_k=10, host_rank=2,
                             staging_rows=12, floor_tokens=FLOOR)


def _rank_floor(table, scratch_run):
    """TP2's cells as the rank computes them (``stage_vram_cells``: the budget
    is the BORN form, the floor stage with every stage row ON), over a bank of
    R + scratch + stage rows at the stage row and the token-cut FA pool at its
    trim cell; the floor per bs (``capture_floors``) under the by-demand form."""
    rb = int(ROW)
    boot = R2 + int(scratch_run)
    bank = dsv.RowTensorGeom(name="bank", rows_boot=boot, rows_max=boot + int(table.rows),
                             row_bytes=rb, alloc_bytes=(boot + int(table.rows)) * rb)
    fa = dsv.KvTensorGeom(geom=dsv.SlotTensorGeom(name="fa", layers=1, slots=TOP,
                                                  slot_bytes=CELLS[2],
                                                  alloc_bytes=TOP * CELLS[2]))
    cells = dsv.stage_vram_cells((), (bank,), (fa,), cap=6, pool_size=6,
                                 extra_max=int(table.rows), stage_tokens=table.tokens,
                                 boot_rows_on=int(table.rows), granule=1 << 16)
    form = dsv.StageForm(tokens=tuple(table.tokens), rows_on=int(table.rows), by_demand=True)
    dsv.check_form_against_cells(form, cells, 6)
    return dsv.capture_floors(form, cells, 6)


@pytest.fixture(autouse=True)
def _no_measured_lru_floor(monkeypatch):
    # the measured-peak LRU floor reads a profile record -- not this question
    monkeypatch.setenv(L.D_KV_STAGE_LRU_FLOOR_ENV, "0")


def test_the_y6n_rank_numbers_reproduce():
    """The fixture is the boot: E 160 - R 67 = 93 ids of 240; the base form
    (stage rows 20 = seat rows 36 - lend head 16, low 8) ran scratch
    58 - (20 - 8) = 46 = LRU 38 + staging 8, and its cells fund 0 rows at the
    top -- the rank's 'C=LRU+staging=46' without a floor suffix."""
    assert ep.step_row_demand(IDS, E[2], R2) == 93
    assert 93 > 2 * (38 + 8)
    t = _tp2_table()
    assert t.low_rows == 8 and (S0 - FLOOR) * CELLS[2] / ROW == pytest.approx(8.96)
    assert (TOP - FLOOR) * CELLS[2] / ROW == pytest.approx(19.2)
    base = dsv.StageForm(tokens=t.tokens, rows_on=36 - LEND_HEAD, by_demand=True)
    floors = _rank_floor(types.SimpleNamespace(rows=base.rows_on, tokens=t.tokens),
                         SCRATCH[2] - (base.rows_on - t.low_rows))
    assert SCRATCH[2] - (base.rows_on - t.low_rows) == 38 + 8
    assert floors == (0, 0, 0, 0, 0, 0)  # D.log 5157: CAPTURE-FLOOR [0, 0, 0, 0, 0, 0]


def test_stage_capture_rows_takes_the_floor_not_the_lend_head(monkeypatch):
    """(a): ``_stage_capture_rows`` = min(seat rows - lend head, floor of the
    captured batch) -- 36 - 16 = 20 rows remain, the 0 is the floor table's.
    With the fixed form the rank's floor at bs 6 is 1."""
    from flliper.srt.layers.moe.expert_offload import MoEExpertOffloadCache

    t = _tp2_table()
    floors = _rank_floor(t, SCRATCH[2] - (int(t.rows) - int(t.low_rows)))
    monkeypatch.setattr(dsv, "capture_floor_rows",
                        lambda n_ids, per_seat: floors[-(-int(n_ids) // 40) - 1])
    fake = types.SimpleNamespace(seat_rows=int(t.rows) + LEND_HEAD, seat_lend_head=LEND_HEAD,
                                 layer=types.SimpleNamespace(top_k=10))
    assert MoEExpertOffloadCache._stage_capture_rows(fake, IDS) == floors[5] >= 1


def test_the_table_top_capacity_is_the_ranks_capture_c():
    """RED on the base: the table priced 47 at the top (58 + 8 - 19), the
    rank's cells give scratch 46 + floor 0 = 46 (its H95 line: C=46, no floor)."""
    t = _tp2_table()
    scratch_run = SCRATCH[2] - (int(t.rows) - int(t.low_rows))
    floors = _rank_floor(t, scratch_run)
    c_rank = scratch_run + floors[5]
    assert t.capacity[5][-1] == c_rank
    # ... with the spare row on top of the honest cut: one row stays ON
    assert floors[5] >= 1
    assert t.stage_rows[-1] == -(-(TOP - FLOOR) * CELLS[2] // ROW) == 20
    assert t.rows == 21


def test_the_launcher_raises_the_derived_wave_cap_the_rank_needs():
    """RED on the base: the launcher let 2 waves pass (Luft 1); the rank needs
    ceil(93 / 46) = 3. Derived cap -> raised through the #251d path."""
    ns = types.SimpleNamespace(env_d=ENV_D, d_pool_waves_derived=True)
    lines = L.apply_d_kv_stage_form(ns, er, _rows(), FORM, _plan(), "D", verify_tokens=4,
                                    top_k=10, kv_token_shares=(0, 40, 24))
    env = L.parse_group_env(ns.env_d)
    t = _tp2_table()
    scratch_run = SCRATCH[2] - (int(t.rows) - int(t.low_rows))
    assert env["FLLIPER_MOE_SCRATCH_SLOTS"].split(",")[2] == str(scratch_run)
    c_rank = scratch_run + _rank_floor(t, scratch_run)[5]
    need = ep.pool_waves_for(IDS, E[2], R2, c_rank)
    assert need == 3
    assert int(env["FLLIPER_OPT_MOE_POOL_OVERFLOW_WAVES"]) >= need
    floor = [ln for ln in lines if "WELLENBODEN JE RANG" in ln]
    assert floor and "rang2 D 93 <= 3 x (45 + 1) = 138" in floor[0]
    assert "BRICHT" not in floor[0]


def test_an_operator_wave_cap_is_refused_by_name_in_the_dry_run():
    """RED on the base (no refusal, the capture died): a cap the operator
    states in --env-d is not the launcher's to raise -- W169 by name."""
    ns = types.SimpleNamespace(env_d=ENV_D, d_pool_waves_derived=False)
    with pytest.raises(L.PdFlipDKvStageWavesRefused, match="W169"):
        L.apply_d_kv_stage_form(ns, er, _rows(), FORM, _plan(), "D", verify_tokens=4,
                                top_k=10, kv_token_shares=(0, 40, 24))


def test_the_wave_floor_reads_the_ranks_scratch_and_floor():
    """The S3g line names the two numbers the rank's H95 line prints: the
    scratch it runs (LRU + staging) and its capture floor."""
    ns = types.SimpleNamespace(env_d=ENV_D, d_pool_waves_derived=True)
    lines = L.apply_d_kv_stage_form(ns, er, _rows(), FORM, _plan(), "D", verify_tokens=4,
                                    top_k=10, kv_token_shares=(0, 40, 24))
    env = L.parse_group_env(ns.env_d)
    run = [int(x) for x in env["FLLIPER_MOE_SCRATCH_SLOTS"].split(",")]
    floor = next(ln for ln in lines if "WELLENBODEN JE RANG" in ln)
    for r in range(3):
        assert "rang%d D " % r in floor and " x (%d + " % run[r] in floor


def test_owned_floor_counts_the_stage_rows_from_the_floor():
    """433a4e01af's owned floor, one formula with the table: the top stage
    takes ceil(19.2) - 8 = 12 rows of TP2's scratch under the floor ladder, not
    ceil(10.24) = 11 -- 93 > 2 x (58 - 12); one row from resident to scratch
    holds it (E - R = 94 <= 2 x 47)."""
    edge = er._EdgeFit(rank=2, local_experts=E[2], scratch_rows=SCRATCH[2],
                       ceiling_max_rows=MAX_ROWS[-1][2], ceiling_fraction=0.5,
                       trim_cell=CELLS[2])
    kw = dict(ids_cap=IDS, waves=2, stage_tokens=S0, stage_row_bytes=ROW,
              stage_floor_tokens=S0 - FLOOR)
    assert er.owned_wave_floor([edge], **kw) == ("rang2 D 93 > 2 x (58 - 12 Stufenzeilen) = 92",)
    assert er.owned_scratch_raise([edge], **kw) == (1,)
    # without a floor ladder: byte-identical to 433a4e01af
    assert er.owned_wave_floor([edge], ids_cap=IDS, waves=2, stage_tokens=S0,
                               stage_row_bytes=ROW) == ()


def test_owned_limits_carry_the_floor_span():
    terms = types.SimpleNamespace(expert_layer_weight_bytes=512 * 2.417 * (1 << 20),
                                  num_experts=512, n_layers=48)
    cfg = {"moe_intermediate_size": 512, "hidden_size": 2048,
           "linear_num_value_heads": 32, "linear_value_head_dim": 128,
           "linear_key_head_dim": 128, "layer_types": ["linear_attention"] * 3}
    kw = dict(text_cfg=cfg, terms=terms, seats=6, kv_tokens=S0, rank_tp_ratio="1,0,0",
              n_ranks=3)
    env = {"FLLIPER_OPT_MOE_POOL_OVERFLOW_WAVES": "2", "FLLIPER_OPT_PDFLIP_D_SEAT_VRAM": "1"}
    assert "stage_floor_tokens" not in er.owned_form_limits(
        dict(env, FLLIPER_PDFLIP_D_KV_STAGE_FLOOR_TOKENS="0"), **kw)
    on = er.owned_form_limits(dict(env, FLLIPER_PDFLIP_D_KV_STAGE_FLOOR_TOKENS="32768"), **kw)
    assert on["stage_floor_tokens"] == S0 - FLOOR


def test_no_floor_ladder_is_byte_identical():
    """Without a floor the stage rows are the plain ceil above S0 (#251c)."""
    t = er.kv_stage_table(_rows(), FORM, kv_cell_bytes=CELLS[2], kv_tokens=S0,
                          local_experts=E[2], verify_tokens=4, top_k=10, host_rank=2,
                          staging_rows=12, floor_tokens=0)
    assert t.tokens == (S0, 393216, TOP)
    assert t.stage_rows == (0, 6, 11) and t.rows == 12 and t.low_rows == 0
