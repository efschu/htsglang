"""#239 S3g floor: the wave floor of EVERY D rank (rc12z30b, 28.09. 20:09:07).

rc12z30b -st (70ac86e2bd, no token cut) died in the decode-graph capture on
TP2: 'Step ids exceed the LRU rows plus the staging rows' -- H95 'captured step
of 240 ids -> 2 wave(s); demand bound min(ids, E-R)=92, C=LRU+staging=45'.
S3g/(d) had given each byteless Form A worker 3 stage rows (scratch 48 -> 45)
for QSA keys a Form A worker never allocates (``qsa_index_on_rank = not
this_rank_is_form_a_worker()``); nothing was born trimmed there ('KV-STAGE ...
born=0'), so the worker built no cells, computed no capture floor and captured
with the rows OFF, while the planner's table counted them ON in S0.

WHAT MUST HOLD.
(1) The planner: a Form A worker's trim cell is its token-cut FA share only;
    a byteless worker gets no stage rows (Form A = #251c's form byte for byte).
(2) ONE formula for the planner and the rank, per D rank r:
    D_r = min(ids, E - R) <= waves x (LRU + staging + capture_on[r]), with the
    scratch AFTER the stage rows moved and capture_on[r] = the rank's capture
    floor where it has KV cells, 0 where it has none. The launcher names each
    rank's air and refuses a broken rank by name (W171) before the boot.
(3) The runtime: a rank with its own stage rows but no KV cells stops by name
    (W-STAGE-ROWS-NO-KV) at the capture and at the wake, never the anonymous
    'Step ids exceed ...'.
"""
from __future__ import annotations

import os
import sys
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.planner import expert_residency as er  # noqa: E402
from flliper.srt.pdflip import d_seat_vram as dsv  # noqa: E402
from flliper.srt.pdflip import launcher as L  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_pdflip_d_kv_stage_launcher_251c as T251  # noqa: E402


def _fit(rank, *, cell, stage=0, experts=193, staging=12):
    return types.SimpleNamespace(rank=rank, kv_cell_bytes=cell, kv_tokens=262144,
                                 local_experts=experts, staging_rows=staging,
                                 kv_stage_cell_bytes=stage)


# ---- (2) the one formula, with the metal numbers of rc12z30b ---------------------------

def test_rc12z30b_metal_numbers_tp2_breaks_tp0_has_one_row_of_air():
    """27B's case: demand [181, 71, 92], scratch after the move [58, 45, 45]
    (ROWS_BY_RANK 33,3,3), 2 waves, TP0 capture floor at bs6 = 33 (floors
    [17, 23, 23, 23, 33, 33]); the workers have no cells, capture_on 0.
    TP2: 2 x 45 = 90 < 92 -- the capture death. TP0: 2 x (58 + 33) = 182,
    air 1."""
    lines, bad = L.rank_wave_floor([181, 71, 92], [58, 45, 45], [33, 0, 0], 2)
    assert bad == [2]
    assert "rang0 D 181 <= 2 x (58 + 33) = 182, Luft 1" in lines[0]
    assert "BRICHT" not in lines[0] and "BRICHT" not in lines[1]
    assert "rang2 D 92 <= 2 x (45 + 0) = 90, Luft -2 -- BRICHT" in lines[2]


def test_rc12z30b_form_with_the_fix_the_workers_keep_their_scratch():
    """With no stage rows on the byteless workers, TP2 captures over 48 rows:
    2 x 48 = 96 >= 92."""
    lines, bad = L.rank_wave_floor([181, 71, 92], [58, 48, 48], [33, 0, 0], 2)
    assert bad == [] and "Luft 4" in lines[2]


# ---- (1) + (2) through the launcher ----------------------------------------------------

def test_byteless_workers_trim_nothing_and_the_launcher_names_every_rank():
    """RED on 70ac86e2bd: the byteless workers got trim cell 768 and 3 stage
    rows; no per-rank floor line existed."""
    fits = [_fit(0, cell=14143), _fit(1, cell=768, experts=120), _fit(2, cell=768, experts=120)]
    assert [er.kv_stage_trim_cell(f) for f in fits] == [14143, 0, 0]
    ns = types.SimpleNamespace(env_d="FLLIPER_MOE_SCRATCH_SLOTS=100,48,48")
    lines = L.apply_d_kv_stage_form(ns, er, T251._rows(), T251.FORM,
                                    types.SimpleNamespace(fits=fits), "D",
                                    verify_tokens=4, top_k=10, kv_token_shares=None)
    env = L.parse_group_env(ns.env_d)
    assert env["FLLIPER_MOE_SCRATCH_SLOTS"] == "67,48,48"
    assert L.D_KV_STAGE_ROWS_BY_RANK_KEY not in env
    floor = [ln for ln in lines if "WELLENBODEN JE RANG" in ln]
    assert len(floor) == 1
    for r in (0, 1, 2):
        assert "rang%d D " % r in floor[0]
    assert "(67 + " in floor[0] and "(48 + 0)" in floor[0]


def test_a_stage_table_on_a_rank_without_kv_is_refused_by_name():
    """A table for a worker whose plan stages no KV (the S3g/(d) form) cannot
    be written: its runtime would build no cells and count the rows OFF."""
    fits = [_fit(0, cell=14143), _fit(1, cell=768, experts=120), _fit(2, cell=768, experts=120)]
    t0 = er.kv_stage_table(T251._rows(), T251.FORM, kv_cell_bytes=14143, kv_tokens=262144,
                           local_experts=193, verify_tokens=4, top_k=10, host_rank=0,
                           staging_rows=12)
    t2 = er.kv_stage_table(T251._rows(), T251.FORM, kv_cell_bytes=768, kv_tokens=262144,
                           local_experts=120, verify_tokens=4, top_k=10, host_rank=2,
                           staging_rows=12)
    group = er.KvStageGroup(tables=(t0, t2), n_ranks=3)
    with pytest.raises(L.PdFlipDKvStageWavesRefused, match=r"W171 .*Stufenzeilen ohne KV"):
        L.kv_stage_wave_floor(group, T251._rows(), fits, group.max_by_seats, 2, "D")


def test_the_cut_form_passes_the_floor_on_every_kv_rank():
    """-st-cut: every KV rank (host, both FA-share workers) passes the floor
    with its own capture floor; the line names all three."""
    FA = 12288
    fits = [_fit(0, cell=1855),
            _fit(1, cell=FA * 3 // 4 + 768, stage=FA * 3 // 4, experts=120),
            _fit(2, cell=FA // 4 + 768, stage=FA // 4, experts=120)]
    ns = types.SimpleNamespace(env_d="FLLIPER_MOE_SCRATCH_SLOTS=100,48,48")
    lines = L.apply_d_kv_stage_form(ns, er, T251._rows(), T251.FORM,
                                    types.SimpleNamespace(fits=fits), "D",
                                    verify_tokens=4, top_k=10, kv_token_shares=(0, 48, 16))
    floor = [ln for ln in lines if "WELLENBODEN JE RANG" in ln]
    assert len(floor) == 1 and "BRICHT" not in floor[0]
    assert all("rang%d D " % r in floor[0] for r in (0, 1, 2))


# ---- (3) the runtime -------------------------------------------------------------------

def _form(rows_on):
    return dsv.StageForm(tokens=(262144, 393216, 524288), rows_on=rows_on,
                         max_by_seats=(2, 1, 1, 1, 0, 0), by_demand=False)


def test_stage_rows_without_kv_cells_stop_by_name():
    """RED on 70ac86e2bd (no such check): TP2 carried 3 stage rows, born=0,
    and died anonymously inside the capture."""
    with pytest.raises(dsv.PdFlipDSeatVramRefused, match="W-STAGE-ROWS-NO-KV"):
        dsv.check_stage_rows_have_kv(_form(3), {}, per_rank=True)


def test_no_rows_or_real_cells_pass():
    dsv.check_stage_rows_have_kv(None, {})
    dsv.check_stage_rows_have_kv(_form(0), {}, per_rank=True)
    dsv.check_stage_rows_have_kv(_form(33), {}, per_rank=False)  # #251c: the host's rows
    dsv.check_stage_rows_have_kv(_form(3), {(1, 0): object()}, per_rank=True)
