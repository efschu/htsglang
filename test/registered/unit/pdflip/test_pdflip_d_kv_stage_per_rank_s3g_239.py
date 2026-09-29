"""#239 S3g/(c): the KV stage form on EVERY rank that holds KV under the token cut.

main 28.09. ~19:10Z: "Bau (c) so: Stufenzeilen je KV-Rang, gleiche Funktion fuer
TP0/Form A und die Worker unter dem Schnitt. Schnitt am Wake ueber die
H95c-Zeilenschaltung, im FRACTION-SOLVE als Residenz gebucht. Ein Test muss
zeigen, dass Form A byte-gleich bleibt (ROWS=33)."

WHAT MUST HOLD.
(1) Form A (no cut, byteless workers): the form is byte-identical to #251c --
    ROWS=33, TP0 scratch 100 -> 67, seat rows 33, no per-rank key.
(2) Every KV rank gets its stage rows from the SAME function
    (``kv_stage_table``) with its own trim cell -- the host's whole cell, a
    worker's token-cut FA share (share x FA cell). A Form A worker holds NO
    QSA keys (``qsa_index_on_rank = not this_rank_is_form_a_worker()``): the
    768 B/token its cell prices are no tensor, so a byteless worker gets no
    stage rows (S3g floor, rc12z30b 28.09. 20:09:07). The launcher
    moves each rank's rows from its scratch into its seat rows, writes
    FLLIPER_PDFLIP_D_KV_STAGE_ROWS_BY_RANK, and the highest stage per seat count
    is the MIN over the KV ranks (a replicated stage choice); the FRACTION-SOLVE
    books the rows as residency, one line per KV rank. Its undo restores all.
(3) The runtime: a KV worker with its own stage rows reads its entry, trims
    its compacted FA pool to S0's compacted rows at birth, does not refuse the stage form; its cells cut its own stage rows at the
    wake (the H95c row switch). A host with a cut share trims its compacted
    FA pool too.
"""
from __future__ import annotations

import os
import sys
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from flliper.srt import rank_role  # noqa: E402
from flliper.srt.planner import expert_residency as er  # noqa: E402
from flliper.srt.pdflip import d_seat_vram as dsv  # noqa: E402
from flliper.srt.pdflip import launcher as L  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_pdflip_d_kv_stage_launcher_251c as T251  # noqa: E402
import test_pdflip_d_kv_stage_worker_251c as W251  # noqa: E402

FA_CELL = 12288  # NF full-attention KV cell, B per token


def _fit(rank, *, cell, stage=0, experts=193, staging=12):
    return types.SimpleNamespace(rank=rank, kv_cell_bytes=cell, kv_tokens=262144,
                                 local_experts=experts, staging_rows=staging,
                                 kv_stage_cell_bytes=stage)


def _cut_plan():
    """The cut [0, 48, 16] of 64: the host keeps its non-FA KV (1855 B/token:
    QSA keys + draft), TP1 holds 3/4 of the FA KV, TP2 1/4."""
    return types.SimpleNamespace(fits=[
        _fit(0, cell=1855),
        _fit(1, cell=FA_CELL * 3 // 4 + 768, stage=FA_CELL * 3 // 4, experts=120),
        _fit(2, cell=FA_CELL // 4 + 768, stage=FA_CELL // 4, experts=120)])


def _form_a_plan():
    """Form A: TP0 holds the whole KV (14143 B/token), the workers only their
    QSA keys (768 B/token) -- the cells the H33 reference measured."""
    return types.SimpleNamespace(fits=[
        _fit(0, cell=14143), _fit(1, cell=768, experts=120), _fit(2, cell=768, experts=120)])


# ---- (1) Form A byte-identical --------------------------------------------------------

def test_form_a_stays_byte_identical_rows_33():
    before = "FLLIPER_MOE_SCRATCH_SLOTS=100,48,48;X=1"
    ns = types.SimpleNamespace(env_d=before)
    lines = L.apply_d_kv_stage_form(ns, er, T251._rows(), T251.FORM, T251._plan(), "D",
                                    verify_tokens=4, top_k=10, kv_token_shares=None)
    env = L.parse_group_env(ns.env_d)
    assert env == {
        "FLLIPER_MOE_SCRATCH_SLOTS": "67,48,48",
        "X": "1",
        "FLLIPER_PDFLIP_D_SEAT_EXPERT_ROWS": "33,0,0",
        "FLLIPER_PDFLIP_D_KV_STAGE_TOKENS": "262144,393216,524288",
        "FLLIPER_PDFLIP_D_KV_STAGE_ROWS": "33",
        "FLLIPER_PDFLIP_D_KV_STAGE_MAX_BY_SEATS": "2,1,1,1,0,0",
    }
    assert "ROWS_BY_RANK" not in ns.env_d
    assert not any("RESIDENZ" in ln or "JE KV-RANG" in ln for ln in lines)


def test_form_a_is_one_table_the_251c_table():
    # the group of one table IS the #251c table
    g = er.kv_stage_group(T251._rows(), T251.FORM, T251._plan().fits, verify_tokens=4, top_k=10)
    t = T251._table()
    assert len(g.tables) == 1 and g.tables[0] == t
    assert g.rows == 33 and g.rows_by_rank == (33, 0, 0)
    assert g.max_by_seats == t.max_by_seats and g.waves == t.waves
    assert g.extra_waves((2,) * 6) == t.extra_waves((2,) * 6)


def test_every_rank_stages_the_kv_it_holds():
    """RED on 70ac86e2bd ([1855, 9984, 3840] / [14143, 768, 768]): a worker's
    trim cell counted QSA keys a Form A worker never allocates."""
    f = _cut_plan().fits
    assert [er.kv_stage_trim_cell(x) for x in f] == [1855, 9216, 3072]
    assert [er.kv_stage_trim_cell(x) for x in _form_a_plan().fits] == [14143, 0, 0]


def test_form_a_byteless_workers_get_no_stage_rows():
    """RED on 70ac86e2bd (ROWS_BY_RANK=33,3,3, scratch 67,45,45): rc12z30b
    (28.09. 20:09:07, -st) gave each byteless worker 3 stage rows for QSA keys
    it never allocates; nothing was born trimmed there ('KV-STAGE ... born=0'),
    the worker computed no capture floor and captured with the rows OFF -- TP2
    C 45 x 2 waves < 92 ids. Form A is #251c's form again, byte for byte."""
    before = "FLLIPER_MOE_SCRATCH_SLOTS=100,48,48;X=1"
    ns = types.SimpleNamespace(env_d=before)
    L.apply_d_kv_stage_form(ns, er, T251._rows(), T251.FORM, _form_a_plan(), "D",
                            verify_tokens=4, top_k=10, kv_token_shares=None)
    env = L.parse_group_env(ns.env_d)
    assert env["FLLIPER_PDFLIP_D_KV_STAGE_ROWS"] == "33"  # TP0 as ever
    assert L.D_KV_STAGE_ROWS_BY_RANK_KEY not in env
    assert env["FLLIPER_MOE_SCRATCH_SLOTS"] == "67,48,48"
    assert env["FLLIPER_PDFLIP_D_SEAT_EXPERT_ROWS"] == "33,0,0"
    assert env["FLLIPER_PDFLIP_D_KV_STAGE_MAX_BY_SEATS"] == "2,1,1,1,0,0"
    g = er.kv_stage_group(T251._rows(), T251.FORM, _form_a_plan().fits, verify_tokens=4,
                          top_k=10)
    assert len(g.tables) == 1 and g.rows_by_rank == (33, 0, 0)
    L.d_kv_stage_undo(ns)
    assert L.parse_group_env(ns.env_d) == L.parse_group_env(before)


# ---- (2) the launcher under the cut --------------------------------------------------

def test_every_kv_rank_gets_its_stage_rows_from_the_same_function():
    """RED before: the launcher wrote no form under a cut with worker KV."""
    before = "FLLIPER_MOE_SCRATCH_SLOTS=100,48,48;X=1"
    ns = types.SimpleNamespace(env_d=before)
    lines = L.apply_d_kv_stage_form(ns, er, T251._rows(), T251.FORM, _cut_plan(), "D",
                                    verify_tokens=4, top_k=10, kv_token_shares=(0, 48, 16))
    env = L.parse_group_env(ns.env_d)
    # the same function per rank: ceil((S2 - S0) x cell / row) + 1
    per_rank = {r: er.kv_stage_table(T251._rows(), T251.FORM, kv_cell_bytes=c,
                                     kv_tokens=262144, local_experts=e, verify_tokens=4,
                                     top_k=10, host_rank=r, staging_rows=12)
                for r, c, e in ((0, 1855, 193), (1, 9216, 120), (2, 3072, 120))}
    assert [per_rank[r].rows for r in (0, 1, 2)] == [6, 22, 8]
    assert env[L.D_KV_STAGE_ROWS_BY_RANK_KEY] == "6,22,8"
    assert env["FLLIPER_PDFLIP_D_KV_STAGE_ROWS"] == "6"
    assert env["FLLIPER_MOE_SCRATCH_SLOTS"] == "94,26,40"
    assert env["FLLIPER_PDFLIP_D_SEAT_EXPERT_ROWS"] == "6,22,8"
    assert env["FLLIPER_PDFLIP_D_KV_STAGE_TOKENS"] == "262144,393216,524288"
    # the stage choice is replicated: the highest stage every KV rank keeps
    want = tuple(min(per_rank[r].max_by_seats[i] for r in (0, 1, 2)) for i in range(6))
    assert env["FLLIPER_PDFLIP_D_KV_STAGE_MAX_BY_SEATS"] == ",".join(str(j) for j in want)
    assert want != per_rank[0].max_by_seats  # a worker binds here, not the host
    # booked as residency in the FRACTION-SOLVE, one line per KV rank
    res = [ln for ln in lines if "D-KV-STUFEN RESIDENZ (#239 S3g)" in ln]
    assert [ln.split("rang")[1].split(":")[0] for ln in res] == ["0", "1", "2"]
    assert "rang1: 22 Stufenzeilen" in res[1] and "Scratch 48 -> 26" in res[1]
    assert "Trim-Zelle 9216 B/Tok" in res[1] and "[11, 21]" in res[1]
    # the wave floor of EVERY rank, named before the boot (S3g floor)
    assert any("WELLENBODEN JE RANG" in ln and "rang2" in ln for ln in lines)
    assert any("JE KV-RANG (#239 S3g)" in ln for ln in lines)
    assert not any("entfaellt" in ln for ln in lines)
    # a second solve pass takes every rank's rows back
    L.d_kv_stage_undo(ns)
    assert L.parse_group_env(ns.env_d) == L.parse_group_env(before)


def test_a_kv_rank_whose_scratch_cannot_fund_its_stages_drops_the_form():
    """A stage one rank cannot fund is a stage no rank may choose."""
    plan = _cut_plan()
    plan.fits[1].staging_rows = 30  # 48 - 22 = 26 <= 30
    ns = types.SimpleNamespace(env_d="FLLIPER_MOE_SCRATCH_SLOTS=100,48,48")
    (line,) = L.apply_d_kv_stage_form(ns, er, T251._rows(), T251.FORM, plan, "D",
                                      verify_tokens=4, top_k=10, kv_token_shares=(0, 48, 16))
    assert "entfaellt" in line and "Rang 1:" in line and "Scratch zu klein" in line
    assert ns.env_d == "FLLIPER_MOE_SCRATCH_SLOTS=100,48,48"


def test_a_cut_the_plan_does_not_carry_per_rank_still_writes_no_form():
    ns = types.SimpleNamespace(env_d="FLLIPER_MOE_SCRATCH_SLOTS=100,48,48")
    (line,) = L.apply_d_kv_stage_form(ns, er, T251._rows(), T251.FORM, T251._plan(), "D",
                                      verify_tokens=4, top_k=10, kv_token_shares="owned")
    assert "entfaellt unter dem Token-Schnitt" in line and "(noch ungeloest)" in line


# ---- (3) the runtime -----------------------------------------------------------------

G = W251.G
PAGE = W251.PAGE


def _worker_armed(tms, rank, bounds, by_rank="6,24,10"):
    from flliper.srt.environ import envs

    ctx = W251._armed(tms, rank)
    ctx.stack += [envs.FLLIPER_PDFLIP_D_KV_STAGE_ROWS_BY_RANK.override(by_rank),
                  mock.patch("flliper.srt.distributed.utils.uneven_dcp_owner_bounds",
                             lambda: bounds)]
    return ctx


def test_a_kv_worker_reads_its_rows_and_trims_its_compacted_fa_pool():
    """RED before: the worker refused the form ('holds full-attention KV') and
    trimmed nothing. TP1 owns 16 of every 64 tokens: its FA pool at the top
    stage (192) is (192 // 64 + 1) x 16 = 64 rows; S0 (64) keeps 32 + page."""
    tms = W251.FakeTms()
    with _worker_armed(tms, rank=1, bounds=(64, 48, 64)):
        assert dsv.stage_form().rows_on == 24
        assert dsv.kv_stage_pool_tokens(10) == 192  # no refusal, the host's rows
        assert dsv.owner_block_for(64) == (64, 16)
        assert dsv.kv_stage_trims_here(64) is True
        assert dsv.kv_stage_trims_here(192) is True  # its QSA keys follow the stage
        assert dsv.kv_stage_trims_here(128) is False  # no stage pool
        assert dsv.kv_stage_boot_rows(64, PAGE) == 32 + PAGE
        t = torch.zeros(64 + PAGE, 256, dtype=torch.int32)
        tms.add(t)
        dsv.kv_stage_born(t, pool_size=64, page_size=PAGE, name="k0")
        (ptr, geom), = dsv._KV_BORN
        assert ptr == t.data_ptr() and geom.owner_block == (64, 16)
        assert geom.slots_for(64) == 32 + PAGE
        assert geom.slots_for(128) == 48 + PAGE and geom.slots_for(192) == 64 + PAGE
        # the mapped prefix is S0's compacted rows, cut at every stage
        assert tms.calls[-1][1] == tuple(dsv.slot_spans(
            geom.geom, 32 + PAGE, G, cuts=[geom.slots_for(tk) for tk in (64, 128, 192)]))
        q = torch.zeros(3, 52 * 64, dtype=torch.int32)
        tms.add(q)
        n_calls = len(tms.calls)
        dsv.kv_stage_born(q, pool_size=192, page_size=PAGE, name="qsa_compressed",
                          tokens_per_slot=4, layers=3, slots=52)
        assert len(tms.calls) == n_calls + 1  # the QSA keys born at S0
        (_, qgeom) = dsv._KV_BORN[-1]
        assert qgeom.owner_block == (0, 0)  # whole context, 4 tokens per slot
        assert tms.calls[-1][1] == tuple(dsv.slot_spans(
            qgeom.geom, qgeom.slots_for(64), G,
            cuts=[qgeom.slots_for(tk) for tk in (64, 128, 192)]))


def test_a_worker_without_its_own_rows_still_refuses_by_name():
    import pytest

    with _worker_armed(W251.FakeTms(), rank=1, bounds=(64, 48, 64), by_rank=""):
        with pytest.raises(dsv.PdFlipDSeatVramRefused, match="holds full-attention KV"):
            dsv.kv_stage_pool_tokens(10)


def test_form_a_ranks_trim_as_before():
    """Byteless worker (ratio 0) without rows: nothing (rc12z11); with its own
    rows: its QSA keys follow the stage; host: the #251c trim."""
    tms = W251.FakeTms()
    with _worker_armed(tms, rank=1, bounds=(64, 0, 0), by_rank=""):
        assert dsv.owner_block_for(192) == (0, 0)
        assert dsv.kv_stage_trims_here(192) is False
    with _worker_armed(tms, rank=1, bounds=(64, 0, 0), by_rank="33,3,3"):
        assert dsv.stage_form().rows_on == 3
        assert dsv.kv_stage_trims_here(192) is True
        assert dsv.kv_stage_pool_tokens(10) == 192  # no refusal: holds no FA KV
    with _worker_armed(tms, rank=0, bounds=None, by_rank=""):
        assert dsv.stage_form().rows_on == 32
        assert dsv.kv_stage_trims_here(192) is True
        assert dsv.kv_stage_boot_rows(192, PAGE) == 64 + PAGE
        assert dsv.stage_mapped_rows(192, 128, PAGE) == 128 + PAGE


def test_a_host_with_a_cut_share_trims_its_compacted_fa_pool():
    with _worker_armed(W251.FakeTms(), rank=0, bounds=(64, 0, 16)):
        assert dsv.stage_form().rows_on == 6
        assert dsv.kv_stage_trims_here(64) is True  # compacted FA pool
        assert dsv.kv_stage_trims_here(192) is True  # QSA keys / draft: whole context
        assert dsv.stage_mapped_rows(64, 128, PAGE) == 48 + PAGE
        assert dsv.stage_mapped_rows(192, 128, PAGE) == 128 + PAGE


def test_the_workers_cells_cut_its_own_stage_rows_at_the_wake():
    """The H95c row switch on a worker: S0 maps its KV prefix and every stage
    row ON; a higher stage maps more KV and turns rows OFF, the total never
    above the boot form."""
    row = dsv.RowTensorGeom("l0.w", 40, 40 + 6, 1 << 20, (46) << 20)
    kv = dsv.KvTensorGeom(dsv.SlotTensorGeom("k0", 1, 64 + PAGE, 1 << 16, (80) << 16),
                          token_pad=PAGE, owner_block=(64, 16))
    cells = dsv.stage_vram_cells([], [row], [kv], cap=2, pool_size=0, extra_max=6,
                                 stage_tokens=(64, 128, 192), boot_rows_on=6, granule=1 << 16)
    ks = [cells[(2, j)].extra_rows for j in range(3)]
    assert ks[0] == 6 and ks[0] > ks[1] > ks[2] >= 0
    for j in range(3):
        c = cells[(2, j)]
        assert c.feasible and c.mapped <= c.cap_mapped
    # each 16 compacted rows x 64 KiB = 1 MiB = one expert row per stage step
    assert ks == [6, 5, 4]
