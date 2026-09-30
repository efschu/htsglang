"""#251c (Nutzer-Order 27.09./28.09. ueber main): KV-Stufen auf dem Form-A-
Attention-Host (TP0) gegen die Experten-Scratch-Zeilen, gewechselt nur am Wake.

  "Leistung geht vor: die Stufe hebt sich nur, wenn der Bedarf des naechsten
   Wakes den KV uebersteigt; zurueck, sobald er passt; juengster parkt erst
   oberhalb der hoechsten Stufe."

WHAT MUST HOLD (the pure half; the runtime wiring has its own tests).
(1) The KV pool is H95c's third post: every (n, j) cell maps at most the BOOT
    form (n = cap, S0, the stage rows ON) -- the form the planner prices.
    Without KV tensors, one stage and no stage rows the cells ARE H95c's k(n).
(2) NF Form A numbers: a +131072-token stage costs ~16 expert rows of 112.5
    MiB at n = cap; seats below the cap fund part of it with GDN pages.
(3) The stage rule is a pure function of the wake request: smallest stage
    holding the demand, rises only above the KV, falls as soon as it fits,
    the top stage + 'over' (the youngest parks) above everything, S0 without
    a demand, and a waves floor (min rows ON) removes stages -- every rank
    that evaluates the same request takes the same stage.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import d_seat_vram as dsv  # noqa: E402

G = 2 << 20
MIB = 1 << 20
PAGE = 64
T0, T1, T2 = 262144, 393216, 524288


def _nf_geometry(rows_boot=103, x=32):
    """x177 GDN pool + the 48-layer bank (Marlin w13/w2, 112.5 MiB a row)."""
    slot = dsv.SlotTensorGeom("gdn_temporal", 36, 39, 48 * 128 * 128 * 2,
                              dsv.align_up(36 * 39 * 48 * 128 * 128 * 2, G))
    rows = []
    for _layer in range(48):
        for rb in (160 * 2560 * 4, 40 * 5120 * 4):
            rows.append(dsv.RowTensorGeom("t", rows_boot, rows_boot + x, rb,
                                          dsv.align_up((rows_boot + x) * rb, G)))
    return slot, rows


def _nf_kv(top=T2):
    """TP0 Form A KV at the top stage: 12 FA layers K and V (512 B/token
    each), the QSA compressed keys (12 layers, 64 B/token/layer at ratio 4)
    and the MTP draft layer -- 14143 B/token as priced (rc12r cell)."""
    rows = top + PAGE
    kv = []
    for _layer in range(12):
        for _kv in ("k", "v"):
            g = dsv.SlotTensorGeom("fa", 1, rows, 512, dsv.align_up(rows * 512, G))
            kv.append(dsv.KvTensorGeom(g, token_ratio=1, token_pad=PAGE))
    cap = -(-rows // 4)
    qsa = dsv.SlotTensorGeom("qsa", 12, cap, 256, dsv.align_up(12 * cap * 256, G))
    kv.append(dsv.KvTensorGeom(qsa, token_ratio=4, token_pad=PAGE))
    rest = 14143 - 12288 - 768  # draft K/V + its indexer share
    for _kv in ("k", "v"):
        g = dsv.SlotTensorGeom("draft", 1, rows, rest // 2, dsv.align_up(rows * (rest // 2), G))
        kv.append(dsv.KvTensorGeom(g, token_ratio=1, token_pad=PAGE))
    return kv


# ---- (1) the third post keeps the boot form's bound ------------------------

def test_without_kv_the_cells_are_h95cs_rows():
    slot, rows = _nf_geometry(rows_boot=120, x=15)
    seat = dsv.seat_vram_rows([slot], rows, cap=6, pool_size=38, extra_max=15, granule=G)
    cells = dsv.stage_vram_cells([slot], rows, (), cap=6, pool_size=38, extra_max=15,
                                 stage_tokens=[T0], boot_rows_on=0, granule=G)
    assert [cells[(n, 0)].extra_rows for n in range(1, 7)] == [r.extra_rows for r in seat]
    assert [cells[(n, 0)].extra_rows for n in range(1, 7)] == [13, 10, 8, 5, 1, 0]
    for r in seat:
        c = cells[(r.n, 0)]
        assert (c.mamba_mapped, c.expert_mapped) == (r.mamba_mapped, r.expert_mapped)


def test_every_cell_maps_at_most_the_boot_form():
    slot, rows = _nf_geometry()
    kv = _nf_kv()
    cells = dsv.stage_vram_cells([slot], rows, kv, cap=6, pool_size=38, extra_max=32,
                                 stage_tokens=[T0, T1, T2], boot_rows_on=32, granule=G)
    boot = cells[(6, 0)]
    assert boot.extra_rows == 32 and boot.mapped == boot.cap_mapped
    for c in cells.values():
        if c.feasible:
            assert c.mapped <= c.cap_mapped, c
    # the KV prefix of a stage is what the stage's tokens need, nothing more
    assert cells[(6, 1)].kv_mapped > boot.kv_mapped
    assert cells[(6, 2)].kv_mapped > cells[(6, 1)].kv_mapped


# ---- (2) the NF numbers ------------------------------------------------------

def test_nf_stage_prices_rows_and_seats_fund_part_of_it():
    slot, rows = _nf_geometry()
    kv = _nf_kv()
    cells = dsv.stage_vram_cells([slot], rows, kv, cap=6, pool_size=38, extra_max=32,
                                 stage_tokens=[T0, T1, T2], boot_rows_on=32, granule=G)
    k0, k1, k2 = (cells[(6, j)].extra_rows for j in range(3))
    # +131072 tokens x 14143 B = 1768 MiB = 16 rows of 112.5 MiB (granule-exact)
    assert 32 - k1 == 16
    assert 32 - k2 == 32 and cells[(6, 2)].feasible
    # fewer seats: the GDN pages they leave pay part of the stage
    assert cells[(1, 1)].extra_rows > k1
    assert cells[(1, 2)].extra_rows > k2
    # a stage the rows cannot pay is infeasible, not negative
    far = dsv.stage_vram_cells([slot], rows, _nf_kv(top=655360), cap=6, pool_size=38,
                               extra_max=32, stage_tokens=[T0, 655360], boot_rows_on=32,
                               granule=G)
    assert not far[(6, 1)].feasible and far[(6, 1)].extra_rows == 0


# ---- (3) the stage rule --------------------------------------------------------

def _cells():
    slot, rows = _nf_geometry()
    return dsv.stage_vram_cells([slot], rows, _nf_kv(), cap=6, pool_size=38, extra_max=32,
                                stage_tokens=[T0, T1, T2], boot_rows_on=32, granule=G)


def test_the_stage_rises_only_above_the_kv_and_falls_when_it_fits():
    cells = _cells()
    assert dsv.choose_stage(cells, 4, 200_000).stage == 0
    assert dsv.choose_stage(cells, 4, T0).stage == 0          # holds exactly: no rise
    assert dsv.choose_stage(cells, 4, T0 + 1).stage == 1      # above S0's KV: rise
    assert dsv.choose_stage(cells, 6, 6 * 60_000).stage == 1  # 6 seats at 60k
    assert dsv.choose_stage(cells, 6, 8 * 60_000).stage == 2
    # the next wake with less demand falls back at once
    assert dsv.choose_stage(cells, 6, 100_000).stage == 0


def test_above_the_top_stage_the_youngest_parks():
    ch = dsv.choose_stage(_cells(), 6, T2 + 10_000)
    assert (ch.stage, ch.over) == (2, True)
    assert not dsv.choose_stage(_cells(), 6, T2).over


def test_no_demand_on_the_wake_is_the_boot_form():
    ch = dsv.choose_stage(_cells(), 3, None)
    assert (ch.stage, ch.tokens, ch.over) == (0, T0, False)


def test_the_waves_floor_removes_stages_that_would_need_more_waves():
    cells = _cells()
    k1 = cells[(6, 1)].extra_rows
    # the captured waves need k1 + 1 rows ON at bs6: S1 and S2 are out
    ch = dsv.choose_stage(cells, 6, T0 + 1, min_rows_on=k1 + 1)
    assert (ch.stage, ch.over) == (0, True)
    # a floor S1 meets keeps S1, S2 stays out
    assert dsv.choose_stage(cells, 6, T2, min_rows_on=k1).stage == 1


def test_every_rank_evaluating_the_same_wake_takes_the_same_stage():
    # three ranks, each with its own cells built from the SAME geometry/table
    # (TP0 has pages, the workers only the replicated table): one answer.
    answers = {dsv.choose_stage(_cells(), n, d).stage
               for _rank in range(3) for n, d in [(5, 333_333)]}
    assert answers == {1}
