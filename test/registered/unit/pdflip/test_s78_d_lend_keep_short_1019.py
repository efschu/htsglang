"""S78 KEEP-SHORT: a short D extend gives back only the lent cells its activation needs.

Metal (rc17 D ``boot_weg2_..._7d93470b5e_1019_171636.D.log``, all three ranks): 381 lends / 312 returns in
45 min, one cycle per short request; after the full return a short mixed extend started with 2.0-2.2 GiB
free (p50, ``PDFLIP-VRAM-PEAK phase=chunk card_free_start_mib``) and grew the allocator by <= 65-437 MiB
(T < 1024); the re-lend after it paused the decodes ~300 ms (``D-DECODE-GAP`` tails with a lend, p50 312).

What must hold (desk rank of test_pdflip_d_seat_live_wipe_s1_0928, lattice step 4 rows, S0 = 32 rows ON,
lend to 40):
(1) all pending work SHORT by the front's R2 verdict -> only the cells above the largest lattice k that
    leaves ``floor + 300 + rate x rows`` MiB free go back; the rows up to k keep their bytes and stay ON;
(2) the card already holds the price -> nothing moves (``keep``);
(3) no cell can stay -> the full return (as before);
(4) big / unpriced / chunked / resumed pending work, a rank without a chunk rate, or the switch off ->
    the full return byte for byte;
(5) before the built extend runs, its EXACT rows price it;
(6) a partial return is topped up by the next lend verdict, a full lend is not re-lent;
(7) danger direction: whatever stays, the card holds the price afterwards.
"""
from __future__ import annotations

import importlib.util
import os
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import d_seat_vram as dsv  # noqa: E402
from flliper.srt.pdflip import d_transient_lend as dtl  # noqa: E402
from flliper.srt.pdflip import extend_trim  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_t_live_wipe_keep_short", os.path.join(os.path.dirname(__file__), "test_pdflip_d_seat_live_wipe_s1_0928.py"))
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)

ROW = 4096
STEP_BYTES = 4 * ROW * 2  # one lattice step (4 rows) over the two layers' banks
MIB = 1 << 20
RATE = "0.5"  # MiB per row (rc17 TP0: 0.5871)
ROWS = 100    # 95 priced + 5 running decodes
NEED = int((0 + 300 + 0.5 * ROWS) * MIB)  # floor 0 + chunk floor 300 + rate x rows


@pytest.fixture(autouse=True)
def _fresh_trim_cache():
    extend_trim.reset_for_tests()
    yield
    extend_trim.reset_for_tests()


def _short(priced=95, halfx=3200):
    return types.SimpleNamespace(rid="pdflip-2-24", output_ids=[], weg2_front_x=6400,
                                 weg2_front_priced=priced, weg2_front_halfx=halfx)


def _live_lent(tms):
    r = h._rank(tms)
    h._pause_all(tms)
    dsv.on_wake(r.sched, types.SimpleNamespace(epoch="e1"), None)
    for p in list(tms.allocs):
        tms.resume(p)
    ctl = r.sched._pdflip_d_seat_vram
    assert ctl.lend(10 * STEP_BYTES) == 8 and ctl.rows_on == 40 and ctl.lent_from == 32
    r.sched.chunked_req = None
    r.sched.running_batch = types.SimpleNamespace(reqs=[1, 2, 3, 4, 5])
    r.sched.waiting_queue = [_short()]
    return r, ctl


def _env(rate=RATE, keep=True, floor="0"):
    stack = [envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override(floor),
             envs.FLLIPER_PDFLIP_D_LEND_KEEP_SHORT.override(keep)]
    if rate is not None:
        stack.append(envs.FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB.override(rate))
    return stack


class _Ctx:
    def __init__(self, tms, free, **kw):
        self.cms = [h._armed(tms)] + _env(**kw) + [mock.patch.object(dtl, "_card_free_bytes", lambda: free)]

    def __enter__(self):
        for c in self.cms:
            c.__enter__()

    def __exit__(self, *a):
        for c in reversed(self.cms):
            c.__exit__(*a)


def _ids_of_rows(tms, bank, rows):
    return tms.ids(bank.data_ptr(), 0, (10 + rows) * ROW)


def test_the_short_pending_rows_are_the_front_priced_sum_plus_the_running_decodes():
    s = types.SimpleNamespace(chunked_req=None, waiting_queue=[_short(95), _short(200)],
                              running_batch=types.SimpleNamespace(reqs=[1, 2, 3]))
    assert dtl.short_pending_rows(s) == 95 + 200 + 3
    for bad in (_short(3200, 3200), types.SimpleNamespace(), _short(halfx=0)):
        s.waiting_queue = [_short(), bad]
        assert dtl.short_pending_rows(s) is None  # (4) one big/unpriced member = the full return
    resumed = _short()
    resumed.output_ids = [7]
    s.waiting_queue = [resumed]
    assert dtl.short_pending_rows(s) is None  # a resumed request carries no first-admission price
    s.waiting_queue, s.chunked_req = [_short()], object()
    assert dtl.short_pending_rows(s) is None
    s.chunked_req, s.waiting_queue = None, []
    assert dtl.short_pending_rows(s) is None


def test_the_price_is_the_chunk_models_inverse():
    with envs.FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB.override("0.5871,0.3,0.31"):
        extend_trim.reset_for_tests()
        assert extend_trim.chunk_need_mib(0, 312) == pytest.approx(300 + 0.5871 * 312)
        assert extend_trim.chunk_need_mib(1, 10) == pytest.approx(303.0)
        # the inverse of rows_cap: the price of the cap's rows fits the post it was cut from
        cap = extend_trim.rows_cap(2283, 0.5871, 64)
        assert extend_trim.chunk_need_mib(0, cap) <= 2283
    extend_trim.reset_for_tests()
    assert extend_trim.chunk_need_mib(0, 100) is None


def test_a_short_extend_keeps_the_cells_the_price_leaves():
    tms = h.CoreTms()
    free = NEED - 20 * 1024  # one 32-KiB step back covers the price
    with _Ctx(tms, free):
        r, ctl = _live_lent(tms)
        kept = {b.data_ptr(): _ids_of_rows(tms, b, 36) for b in r.banks}
        assert dtl.round_start(r.sched) == "return"
        assert ctl.rows_on == 36 and ctl.lent_from == 32  # (1) the lend stays open above 32
        assert ctl.returned_rows == 4
        for b in r.banks:
            assert _ids_of_rows(tms, b, 36) == kept[b.data_ptr()]  # the kept rows keep their bytes
            m = next(m for m in ctl.row_tensors if m.ptr == b.data_ptr())
            assert tms.mapped(b.data_ptr()) == dsv.span_bytes(ctl.bank_spans(m, 36))
        for c in r.caches:
            assert int(c._pool_tables.seat_on) == 36
        released = ctl._bank_bytes_at(40) - ctl._bank_bytes_at(36)
        assert free + released >= NEED  # (7)


def test_the_lend_floor_is_part_of_the_price():
    """The kept lend must leave the near-OOM edge AND the extend's price: free = price without the floor
    is one floor short, so a step goes back."""
    tms = h.CoreTms()
    with _Ctx(tms, NEED, floor="1"):
        r, ctl = _live_lent(tms)
        assert dtl.round_start(r.sched) == "return" and ctl.rows_on < 40
        released = ctl._bank_bytes_at(40) - ctl._bank_bytes_at(ctl.rows_on)
        assert NEED + released >= NEED + MIB or ctl.lent_from is None


def test_the_card_holding_the_price_moves_nothing():
    tms = h.CoreTms()
    with _Ctx(tms, NEED):
        r, ctl = _live_lent(tms)
        mapped = {b.data_ptr(): tms.mapped(b.data_ptr()) for b in r.banks}
        assert dtl.round_start(r.sched) == "keep"  # (2)
        assert ctl.rows_on == 40 and ctl.lent_from == 32 and ctl.returned_rows == 0
        assert {b.data_ptr(): tms.mapped(b.data_ptr()) for b in r.banks} == mapped


def test_no_cell_can_stay_is_the_full_return():
    tms = h.CoreTms()
    with _Ctx(tms, NEED - 40 * 1024):  # two steps would be needed, the second is the phase's own
        r, ctl = _live_lent(tms)
        assert dtl.round_start(r.sched) == "return"
        assert ctl.rows_on == 32 and ctl.lent_from is None  # (3)


@pytest.mark.parametrize("case", ["big", "unpriced", "chunked", "no_rate", "switch_off"])
def test_everything_else_is_the_full_return(case):
    tms = h.CoreTms()
    kw = {"rate": None} if case == "no_rate" else {"keep": False} if case == "switch_off" else {}
    with _Ctx(tms, NEED, **kw):
        r, ctl = _live_lent(tms)
        if case == "big":
            r.sched.waiting_queue = [_short(priced=4000)]
        elif case == "unpriced":
            r.sched.waiting_queue = [object()]
        elif case == "chunked":
            r.sched.chunked_req = object()
        assert dtl.round_start(r.sched) == "return"
        assert ctl.rows_on == 32 and ctl.lent_from is None  # (4)


def test_the_built_extend_is_priced_with_its_exact_rows():
    from flliper.srt.model_executor.forward_batch_info import ForwardMode

    tms = h.CoreTms()
    with _Ctx(tms, NEED):
        r, ctl = _live_lent(tms)
        assert dtl.round_start(r.sched) == "keep"
        r.sched.waiting_queue = []
        small = types.SimpleNamespace(forward_mode=ForwardMode.MIXED, extend_num_tokens=ROWS)
        assert dtl.on_batch(r.sched, small) == "keep" and ctl.rows_on == 40
        # (5) a built extend priced above the card: the cells go back until the price fits
        wide = types.SimpleNamespace(forward_mode=ForwardMode.EXTEND, extend_num_tokens=ROWS + 60)
        need_wide = int((300 + 0.5 * (ROWS + 60)) * MIB)
        assert need_wide > NEED
        assert dtl.on_batch(r.sched, wide) == "return"
        assert ctl.rows_on < 40
        assert NEED + ctl._bank_bytes_at(40) - ctl._bank_bytes_at(ctl.rows_on) >= need_wide \
            or ctl.lent_from is None


def test_a_partial_return_is_topped_up_and_a_full_lend_is_not_relent():
    from flliper.srt.model_executor.forward_batch_info import ForwardMode

    decode = types.SimpleNamespace(forward_mode=ForwardMode.DECODE)
    tms = h.CoreTms()
    with _Ctx(tms, NEED - 20 * 1024), envs.FLLIPER_PDFLIP_D_LEND_SETTLE_ROUNDS.override(1):
        r, ctl = _live_lent(tms)
        assert dtl.round_start(r.sched) == "return" and ctl.rows_on == 36
        r.sched.waiting_queue = []
        with mock.patch.object(dtl, "_card_free_bytes", lambda: 10 * STEP_BYTES):
            assert dtl.on_batch(r.sched, decode) == "lend"  # (6) the difference is mapped again
            assert ctl.rows_on == 40 and ctl.lent_from == 32
            assert dtl.on_batch(r.sched, decode) is None    # a full lend stays as it is


def test_the_decode_gap_hears_the_lend_moves():
    tms = h.CoreTms()
    with _Ctx(tms, NEED - 20 * 1024), mock.patch.object(dtl, "_gap_note") as note:
        r, _ctl = _live_lent(tms)
        dtl.round_start(r.sched)
        # S106: the move's own host wall rides with the note (D-DECODE-GAP return_ms)
        args = note.call_args[0]
        assert args[:2] == (r.sched, "lend_return") and isinstance(args[2], float) and args[2] >= 0.0


def test_review_a3_the_price_uses_the_effective_rate_not_the_start_rate(monkeypatch):
    """Under-pricing is the danger direction: with the measurement armed, the vote cuts with
    max(start, measured x RATE_SAFETY) -- the price must use the same rate (rc17 TP2 0.6308 vs start 0.5871)."""
    with envs.FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB.override("0.5871"):
        extend_trim.reset_for_tests()
        monkeypatch.setattr(extend_trim, "measure_armed", lambda rank=None: True)
        extend_trim._CACHE["measured"] = 0.5485  # x 1.15 = 0.6308 > 0.5871
        eff = extend_trim.effective_rate(0.5871, 2)
        assert eff > 0.5871
        assert extend_trim.chunk_need_mib(2, 1000) == pytest.approx(300 + eff * 1000)
        assert extend_trim.chunk_need_mib(2, 1000) > 300 + 0.5871 * 1000


def test_review_a2_a_waiting_short_request_logs_keep_once(caplog):
    """A short request waiting for a seat re-prices every iteration; only a change of (rows_on, rows,
    need_mib) is a line (review: 50 iterations gave 100 lines before)."""
    import logging

    tms = h.CoreTms()
    with _Ctx(tms, NEED + 64 * MIB), caplog.at_level(logging.INFO, logger=dtl.__name__):
        r, ctl = _live_lent(tms)
        for _ in range(50):
            assert dtl.round_start(r.sched) == "keep"
        keeps = [m for m in caplog.messages if " keep rows_on " in m]
        assert len(keeps) == 1, keeps
        r.sched.waiting_queue = [_short(), _short(40)]  # the price changed: one new line
        assert dtl.round_start(r.sched) == "keep"
        assert len([m for m in caplog.messages if " keep rows_on " in m]) == 2
        assert ctl.rows_on == 40


def test_review_h1_a_full_lend_below_x_max_is_not_relent():
    """The one-lend-per-settle lock (no partial return before) holds even with headroom above the lend."""
    from flliper.srt.model_executor.forward_batch_info import ForwardMode

    decode = types.SimpleNamespace(forward_mode=ForwardMode.DECODE)
    tms = h.CoreTms()
    with _Ctx(tms, NEED), envs.FLLIPER_PDFLIP_D_LEND_SETTLE_ROUNDS.override(1):
        r = h._rank(tms)
        h._pause_all(tms)
        dsv.on_wake(r.sched, types.SimpleNamespace(epoch="e1"), None)
        for p in list(tms.allocs):
            tms.resume(p)
        ctl = r.sched._pdflip_d_seat_vram
        r.sched.waiting_queue, r.sched.chunked_req = [], None
        assert ctl.lend(STEP_BYTES) == 4 and ctl.rows_on == 36  # a full lend of what the card had then
        with mock.patch.object(dtl, "_card_free_bytes", lambda: 10 * STEP_BYTES):
            assert dtl.on_batch(r.sched, decode) is None and ctl.rows_on == 36
