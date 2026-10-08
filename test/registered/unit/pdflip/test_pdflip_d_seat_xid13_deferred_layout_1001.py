"""xid13 kvh (01.10., boot dauer10011653, D TP0 = 5090): Xid 13 MMU fault ~5 s
/ ~100 bs1 decode rounds after the first live KV-stage change S0->S1.

D.log 17:03:20 TP0, in this order:
  PDFLIP-RESUME expert-rearm ... deferred=116        (H31b: the FULL layout is
                                                    built here, 74 seat rows ON)
  D-SEAT-VRAM LIVE-SPANS stage=S1 rows_on 74->70    (rows 70..73 OFF, pages
                                                    released, bank-first)
  PDFLIP-REARM-DEFER landed layers=29                 (the 74-row layout lands)

The landing wrote rows 70..73 back as FREE (key -1, use 0). A free row is the
step's first victim; once the LRU below had filled, a miss took row 70 and
pool_copy stored into its unmapped tail. cuda-gdb on the lightweight core
(cudacore_..._1979): pool_copy, STG to dst row base 0x35e100000, row 0x190000
B, the mapping ends at row base + 0x180000 -- row_spans' 2 MiB outward
rounding of rows_boot + 70, i.e. the FIRST OFF seat row.

The invariants replayed here: every row the step can write lies in the mapped
row prefix, every LRU row the tables count is < seat_base + rows_on, no table
names a row in the released cells."""

import logging

import torch

from flliper.srt.layers.moe import expert_pool_device as epd
from flliper.srt.pdflip.d_seat_vram import RowTensorGeom, row_spans

E, R, S, C, X = 160, 8, 4, 24, 80
BASE = R + C  # seat_base = rows_boot
ROW_BYTES = 0x190000  # the faulting row (cudacore_1979)
GRANULE = 2 << 20


def _host_row():
    return [-1 if e < R else e - R for e in range(E)]


def _tables():
    return epd.allocate_pool_tables(
        "cpu", E, R + C + X, R, S, {e: e for e in range(R)}, _host_row(), seat_rows=X)


def _rearm_shrink_land(built_on=74, live_on=70):
    """wake (rows ON) -> rearm builds the full layout -> live stage change ->
    the deferred layout lands (DeferredRowsFill._promote)."""
    t = _tables()
    epd.set_seat_rows_on(t, built_on, device_write=True)
    full = epd.pool_layout_tensors(t, {e: e for e in range(R)}, _host_row())
    epd.set_seat_rows_on(t, live_on, device_write=True)
    epd.apply_pool_layout(t, full)
    return t, full


def _mapped_end(rows_on):
    g = RowTensorGeom(name="w13", rows_boot=BASE, rows_max=BASE + X,
                      row_bytes=ROW_BYTES, alloc_bytes=(BASE + X) * ROW_BYTES)
    return row_spans(g, BASE + rows_on, GRANULE)[-1][1]


def _decode(t, rounds=60, per_round=4):
    """bs1 rounds of fresh non-resident experts: every miss takes a victim;
    returns every bank row the copies wrote."""
    buf = epd.allocate_step_buffers("cpu", E)
    dsts = []
    nxt = R
    for _ in range(rounds):
        ids = []
        for _k in range(per_round):
            ids.append(nxt)
            nxt = R + (nxt - R + 1) % (E - R)
        pairs, _ = epd.step_reference(t, torch.tensor(ids, dtype=torch.int32), buf)
        dsts.extend(d for _s, d in pairs)
    return dsts


def test_a_layout_built_before_a_live_shrink_lands_with_the_off_rows_off():
    t, _full = _rearm_shrink_land()
    assert epd.seat_off_range(t) == (BASE + 70, BASE + X)
    key = t.row_key[BASE + 70:BASE + X]
    use = t.row_use[BASE + 70:BASE + X]
    assert bool((key == epd.SEAT_OFF_KEY).all()), key[:6].tolist()
    assert bool((use == epd.ROW_USE_NEVER).all())
    # every LRU row the tables can hand out lies below seat_base + rows_on
    free = [r for r in range(t.lru_start, t.pool_rows) if int(t.row_key[r]) == -1]
    assert max(free) < BASE + 70


def test_no_copy_after_the_landing_writes_past_the_mapped_row_prefix():
    t, _full = _rearm_shrink_land()
    end = _mapped_end(70)
    # the metal geometry: the first OFF row is mapped only partly
    assert (BASE + 70) * ROW_BYTES < end < (BASE + 71) * ROW_BYTES
    dsts = _decode(t)
    assert len(dsts) > (C - S) + 70, "the replay must exhaust the LRU rows"
    bad = sorted({d for d in dsts if (d + 1) * ROW_BYTES > end})
    assert not bad, f"copies into released cells: rows {bad[:4]} (first OFF row {BASE + 70})"
    assert max(dsts) < BASE + 70
    assert int(t.error[0]) == 0


def test_a_grow_before_the_landing_keeps_the_new_rows_on():
    t, _full = _rearm_shrink_land(built_on=70, live_on=74)
    assert bool((t.row_key[BASE + 70:BASE + 74] == -1).all())
    assert bool((t.row_use[BASE + 70:BASE + 74] == 0).all())
    assert bool((t.row_key[BASE + 74:BASE + X] == epd.SEAT_OFF_KEY).all())
    assert epd.pool_row_capacity(t) == (C - S) + S + 74


def test_the_landing_is_the_reinit_of_the_live_count():
    t, _full = _rearm_shrink_land()
    ref = _tables()
    epd.set_seat_rows_on(ref, 70, device_write=True)
    epd.reinit_pool_tables(ref, {e: e for e in range(R)}, _host_row())
    for name in ("row_key", "row_use", "hot_phys", "pf_row", "host_row", "staging_rows"):
        assert torch.equal(getattr(t, name), getattr(ref, name)), name


def test_without_seat_rows_the_layout_is_byte_identical():
    host = [-1 if e < R else e - R for e in range(E)]
    t = epd.allocate_pool_tables("cpu", E, R + C, R, S, {e: e for e in range(R)}, host)
    ref = epd.allocate_pool_tables("cpu", E, R + C, R, S, {e: e for e in range(R)}, host)
    lay = epd.pool_layout_tensors(t, {e: e for e in range(R)}, host)
    assert getattr(lay, "seat_on", None) is None
    t.row_key[R + 1] = 9
    t.hot_phys[9] = R + 1
    epd.apply_pool_layout(t, lay)
    for name in ("row_key", "row_use", "hot_phys", "pf_row"):
        assert torch.equal(getattr(t, name), getattr(ref, name)), name


def test_the_deferred_fill_names_the_restamp(caplog):
    from flliper.srt.layers.moe.expert_offload import (
        DeferredRows,
        DeferredRowsFill,
        MoEExpertOffloadCache,
    )

    t = _tables()
    epd.set_seat_rows_on(t, 74, device_write=True)
    full = epd.pool_layout_tensors(t, {e: e for e in range(R)}, _host_row())
    epd.set_seat_rows_on(t, 70, device_write=True)
    cache = object.__new__(MoEExpertOffloadCache)
    cache._pool_tables = t
    cache._scratch_holds = {}
    cache._pool_pf_buffers = None
    cache._deferred_rows = DeferredRows(entries=[], runs=(), full=full, rows=0, layer_id=3)
    fill = DeferredRowsFill(stream_ops=None)
    fill.add(cache)
    with caplog.at_level(logging.INFO):
        fill.land(cache)
    assert cache._deferred_rows is None
    assert bool((t.row_key[BASE + 70:BASE + X] == epd.SEAT_OFF_KEY).all())
    assert any("PDFLIP-REARM-DEFER SEAT-RESTAMP layers=1 rows_on 74->70" in r.getMessage()
               for r in caplog.records)
