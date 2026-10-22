"""LEND-COST (21.10., nf-rc30-tp0lend-1021): the D-TRANSIENT-LEND short-extend price follows the MEASURED card
cost of the extend's row class -- planner order: no reserve, no safety factor, the 300 MiB HALT line untouched.

Metal (NF rc30 D ``boot_weg2_..._1d822bd150_1021_063252.D.log``): TP0 (5090) is a START-RATE rank (its profile
record 0.5871 is kept, Q-694b EXTEND-RATE is armed on TP1/TP2 only), so review A kept its S78 SUM
``767 + 300 + 0.5871 x rows`` -- 1069 MiB for a 5-row extend (``need_mib=1069 card_free_mib=780``, 06:36:55):
1223 returns in 2 h 13 min (133.9 s host, a decode pause each), TP1/TP2 287 / 277; the TP0 target extends of
<= 64 rows grew the allocator by at most 449 MiB (PDFLIP-VRAM-PEAK transient_mib).

What must hold (S78 desk rank = TP0, lattice step 4 rows, S0 = 32 rows ON, lend to 40):
(1) REPRO: the 5-row extend on TP0 with 780 MiB free returns rows today (no reading: the sum); after the
    rank measured its <= 64-row class at 449 MiB it keeps them (price max(767, 302.9, 749) = 767);
(2) DANGER (OOM): the price is never under ``300 + measured class maximum`` -- one MiB short = a return;
    whatever stays, the card holds the price;
(3) an unmeasured class keeps the old price (TP0: the sum) -- nothing is guessed from another class;
(4) the reading = Q-694b's target reading (max(peak - start allocated, reserved growth)) + the draft extend's
    reserved growth, closed at the next pass BEFORE any lend move; class maximum, ratchet up only; a class
    prices only up to the widest extend measured in it (no extrapolation inside a doubling class);
(5) armed on a rank WITH a lend floor (D) even where Q-694b is not (TP0); the rate path stays untouched;
(6) the HALT instrument: an extend batch that ends under 300 MiB is one named line (lend open or not);
(7) switch off = the old price and no reading on TP0, byte for byte;
(8) a measured-rate rank (TP1/TP2): the measured term only ever RAISES the max price;
(9) rank-unity: the verdict (short or not, return/lend/hold) never reads the measured cost.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import d_transient_lend as dtl  # noqa: E402
from flliper.srt.pdflip import extend_trim  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_t_keep_short_for_lend_cost", os.path.join(os.path.dirname(__file__), "test_s78_d_lend_keep_short_1019.py"))
s78 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s78)
h = s78.h

MIB = 1 << 20
FLOORS = "767,700,701"
RATE = "0.5871"
SUM_5 = 767 + 300 + 0.5871 * 5          # 1069.9 -- the metal need_mib=1069


@pytest.fixture(autouse=True)
def _fresh():
    extend_trim.reset_for_tests()
    with envs.FLLIPER_PDFLIP_EXTEND_RATE_MEASURE.override(True), \
            envs.FLLIPER_PDFLIP_EXTEND_RATE_MEASURE_RANKS.override("1,2"):
        yield
    extend_trim.reset_for_tests()


# ---- fakes: the allocator / driver readings of one extend batch ---------------------------------------

class Cuda:
    def __init__(self, peak, alloc, reserved, free_mib=2000):
        self.peak, self.alloc, self.reserved, self.free_mib = peak, alloc, reserved, free_mib

    def memory_stats(self):
        return {"allocated_bytes.all.peak": int(self.peak * MIB),
                "allocated_bytes.all.current": int(self.alloc * MIB),
                "reserved_bytes.all.current": int(self.reserved * MIB)}

    def mem_get_info(self):
        return int(self.free_mib * MIB), 32088 * MIB

    def is_current_stream_capturing(self):
        return False


class _Mode:
    def is_extend(self):
        return True

    def is_target_verify(self):
        return False


def _fb(rows):
    return types.SimpleNamespace(forward_mode=_Mode(), input_ids=types.SimpleNamespace(shape=(rows,)))


def _extend(rows, transient, growth=0, draft=0, rank=0, free_end=2000):
    """One extend batch on this rank: open after the trim, target close, draft growth, next-pass settle."""
    runner = types.SimpleNamespace(tp_rank=rank)
    worker = types.SimpleNamespace(tp_rank=rank, model_runner=runner)
    cuda = Cuda(peak=30000, alloc=30000, reserved=31000)
    extend_trim.measure_open(worker, _fb(rows), cuda)
    cuda.peak, cuda.reserved = 30000 + transient, 31000 + growth
    extend_trim.measure_close(runner, _fb(rows), cuda, rank=rank)
    cuda.reserved += draft
    cuda.free_mib = free_end
    return extend_trim.lend_cost_settle(rank, cuda)


def _env(keep=True):
    return [envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override(FLOORS),
            envs.FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB.override(RATE),
            envs.FLLIPER_PDFLIP_D_LEND_KEEP_SHORT.override(keep)]


class _Ctx:
    """The S78 desk rank (TP0) with the metal floors/rate and a fixed card reading."""

    def __init__(self, tms, free_mib):
        self.cms = [h._armed(tms)] + _env() + [
            mock.patch.object(dtl, "_card_free_bytes", lambda: int(free_mib * MIB))]

    def __enter__(self):
        for c in self.cms:
            c.__enter__()

    def __exit__(self, *a):
        for c in reversed(self.cms):
            c.__exit__(*a)


def _tp0_five_rows(tms):
    """The metal queue: one short request (priced 1 < halbX) + 4 running decodes = 5 rows."""
    r, ctl = s78._live_lent(tms)
    r.sched.waiting_queue = [s78._short(priced=1)]
    r.sched.running_batch = types.SimpleNamespace(reqs=[1, 2, 3, 4])
    assert dtl.short_pending_rows(r.sched) == 5
    return r, ctl


def _round_start_at(free_mib):
    tms = h.CoreTms()
    with _Ctx(tms, free_mib):
        r, ctl = _tp0_five_rows(tms)
        out = dtl.round_start(r.sched)
        released = ctl._bank_bytes_at(40) - ctl._bank_bytes_at(ctl.rows_on)
        return out, ctl, released


# ---- (1) the repro ------------------------------------------------------------------------------------

def test_repro_tp0_five_row_extend_returns_today_and_keeps_after_the_measurement():
    with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override(FLOORS), \
            envs.FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB.override(RATE):
        assert not dtl.rate_measured(0)                                   # TP0: start rate, review A's sum
        assert dtl.short_extend_need_mib(0, 5) == pytest.approx(SUM_5)    # need_mib=1069
    out, ctl, _ = _round_start_at(780)                                    # card_free_mib=780 (06:36:55)
    assert out == "return" and ctl.rows_on < 40 and ctl.returned_rows > 0
    # the rank measures its <= 64-row class: the metal maximum, 449 MiB
    with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override(FLOORS):
        line = _extend(5, transient=449)
    assert "EXTEND-LEND-COST rank=0 rows=5 class=64 cost_mib=449" in line
    with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override(FLOORS), \
            envs.FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB.override(RATE):
        assert dtl.short_extend_need_mib(0, 5) == pytest.approx(767.0)  # max(767, 302.9, 749)
    out, ctl, _ = _round_start_at(780)
    assert out == "keep" and ctl.rows_on == 40 and ctl.returned_rows == 0
    # the built extend's exact rows price the same way
    tms = h.CoreTms()
    with _Ctx(tms, 780):
        r, ctl = _tp0_five_rows(tms)
        batch = types.SimpleNamespace(forward_mode=types.SimpleNamespace(
            is_decode=lambda: False, is_target_verify=lambda: False, is_idle=lambda: False,
            is_extend=lambda: True), extend_num_tokens=5)
        assert dtl.on_batch(r.sched, batch) == "keep" and ctl.rows_on == 40


# ---- (2) danger: never under 300 + the measured class maximum ----------------------------------------

@pytest.mark.parametrize("cost", [0, 449, 466, 467, 480, 481, 900])
def test_danger_the_price_is_never_under_300_plus_the_measured_cost(cost):
    with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override(FLOORS), \
            envs.FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB.override(RATE):
        _extend(5, transient=cost)
        need = dtl.short_extend_need_mib(0, 5)
        if cost > 0:
            assert need == pytest.approx(max(767.0, 300 + 0.5871 * 5, 300.0 + cost))
            assert need >= 300.0 + cost
        else:
            assert need == pytest.approx(767.0)
    if cost > 0:
        # one MiB under the price never keeps the whole lend; whatever stays, the card holds the price
        price = max(767.0, 300.0 + cost)
        out, ctl, released = _round_start_at(price - 1)
        assert out == "return" and ctl.rows_on < 40
        assert ctl.lent_from is None or (price - 1) * MIB + released >= price * MIB
        out, ctl, _ = _round_start_at(price)
        assert out == "keep" and ctl.rows_on == 40


# ---- (3) an unmeasured class keeps the old price -------------------------------------------------------

def test_an_unmeasured_class_keeps_the_sum_nothing_is_borrowed_from_another_class():
    with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override(FLOORS), \
            envs.FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB.override(RATE):
        _extend(42, transient=106)                                        # class 64 measured up to 42 rows
        assert dtl.short_extend_need_mib(0, 42) == pytest.approx(767.0)
        assert dtl.short_extend_need_mib(0, 43) == pytest.approx(767 + 300 + 0.5871 * 43)   # wider than measured
        assert dtl.short_extend_need_mib(0, 65) == pytest.approx(767 + 300 + 0.5871 * 65)   # class 128: none
        assert dtl.short_extend_need_mib(0, 1285) == pytest.approx(767 + 300 + 0.5871 * 1285)
        _extend(1500, transient=1063)                                     # class 2048 measured
        assert dtl.short_extend_need_mib(0, 1285) == pytest.approx(max(767, 300 + 0.5871 * 1285, 1363))
        assert dtl.short_extend_need_mib(0, 200) == pytest.approx(767 + 300 + 0.5871 * 200)


def test_the_width_vote_stays_funded_the_chunk_price_stays_in_the_max():
    """A measured class under the chunk model's own price: the short request must still run in ONE chunk
    (``free >= 300 + rate x rows`` = ``rows_cap >= rows``) -- the measured term never lowers it."""
    with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override(FLOORS), \
            envs.FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB.override(RATE):
        _extend(2000, transient=1000)                                     # class 2048: 1300 < 300 + 0.5871 x 2000
        need = dtl.short_extend_need_mib(0, 2000)
        assert need == pytest.approx(300 + 0.5871 * 2000)
        assert extend_trim.rows_cap(need, 0.5871, 64) >= 1984


def test_danger_a_class_never_prices_an_extend_wider_than_its_widest_reading():
    """A class doubles its rows: TP0 spends ~0.75 MiB/row at 2139 rows (review A, rc28 _201023), so a
    2100-row reading (1300 MiB) must not price a 4000-row extend (the old price holds until one is read)."""
    with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override(FLOORS), \
            envs.FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB.override(RATE):
        _extend(2100, transient=1300)
        # EXTEND-KORRIDOR (22.10.): a reading of >= 2048 rows also prices the rate (1300 / 2100 = 0.6191 MiB/row,
        # extend_trim.lend_rate) -- the class still never prices wider than its widest reading
        lr = 0.6191
        assert dtl.short_extend_need_mib(0, 2100) == pytest.approx(max(767, 300 + lr * 2100, 1600))
        assert dtl.short_extend_need_mib(0, 4000) == pytest.approx(767 + 300 + lr * 4000)
        _extend(4000, transient=2900)
        assert dtl.short_extend_need_mib(0, 4000) == pytest.approx(3200.0)
        assert dtl.short_extend_need_mib(0, 2100) == pytest.approx(3200.0)   # the class maximum


def test_the_row_classes():
    assert [extend_trim.lend_cost_class(r) for r in (1, 5, 64, 65, 128, 129, 256, 1285, 2048, 2049)] == \
        [64, 64, 64, 128, 128, 256, 256, 2048, 2048, 4096]


# ---- (4) the reading ------------------------------------------------------------------------------------

def test_the_reading_is_the_target_cost_plus_the_draft_growth_and_ratchets_up():
    with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override(FLOORS):
        line = _extend(42, transient=106, growth=118, draft=12)
        assert "cost_mib=130" in line and "forward_mib=118" in line and "draft_growth_mib=12" in line
        assert extend_trim.lend_cost_mib(5) == 130.0
        assert _extend(10, transient=90) is None                          # lower: no line, no change
        assert extend_trim.lend_cost_mib(5) == 130.0
        line = _extend(12, transient=449)
        assert "cost_mib=449" in line and "prev=130" in line and "widest=42" in line
        assert extend_trim.lend_cost_mib(42) == 449.0
        assert extend_trim.lend_cost_mib(43) is None                      # wider than any measured one
        assert _extend(60, transient=10) is None                          # wider, cheaper: the reach grows
        assert extend_trim.lend_cost_mib(60) == 449.0
        assert extend_trim.lend_cost_mib(65) is None


def test_the_tail_closes_once_and_an_inherited_peak_is_the_bound_never_a_measured_0():
    """A1 (review 21.10.): on a backend without a peak reset the open cannot re-base the counter; an extend that
    does not raise the inherited peak (500 MiB over the start allocation) books that peak as its bound, not 0."""
    with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override(FLOORS):
        runner = types.SimpleNamespace(tp_rank=0)
        worker = types.SimpleNamespace(tp_rank=0, model_runner=runner)
        cuda = Cuda(peak=30500, alloc=30000, reserved=31000)              # a peak from before the open
        extend_trim.measure_open(worker, _fb(5), cuda)
        extend_trim.measure_close(runner, _fb(5), cuda, rank=0)           # nothing raised, nothing grew
        assert extend_trim._CACHE["lend_cost_tail"] is not None
        extend_trim.lend_cost_settle(0, cuda)
        assert extend_trim.lend_cost_mib(5) == 500.0
        assert extend_trim.lend_cost_settle(0, cuda) is None              # closed once
        assert extend_trim._CACHE["lend_cost_tail"] is None


class ResettingCuda(Cuda):
    """A1/B1: a backend with ``reset_peak_memory_stats`` (the peak counter restarts at the allocation now)."""

    def reset_peak_memory_stats(self):
        self.peak = self.alloc


def _vpw_state():
    from flliper.srt.model_executor import vram_peak_window as vpw

    return vpw


def test_a1_the_open_closes_the_round_window_and_the_extend_reads_its_own_peak(caplog):
    """A1 + B1: decode rounds left the counter 400 MiB over the allocation; the extend itself peaks at 380.
    The open closes and prints the round window (decode only), re-bases, and the extend reads 380 -- not 0
    (masked), not 400 (the decode peak). The forward-end chunk record then holds the extend's own transient."""
    vpw = _vpw_state()
    saved = dict(vpw._STATE)
    try:
        with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override(FLOORS), envs.FLLIPER_PDFLIP_VRAM_PEAK.override(True), \
                mock.patch("flliper.srt.pdflip.vram_actual.on_window", lambda *a, **k: False), \
                caplog.at_level(logging.INFO):
            vpw._STATE.update(t0=1.0, start_alloc=30000 * MIB, free_start=2000 * MIB, rounds=12,
                              retries0=0, ooms0=0, seats=1)
            runner = types.SimpleNamespace(tp_rank=0)
            worker = types.SimpleNamespace(tp_rank=0, model_runner=runner)
            cuda = ResettingCuda(peak=30400, alloc=30000, reserved=31000)
            extend_trim.measure_open(worker, _fb(1100), cuda)
            rounds = [r.getMessage() for r in caplog.records if "phase=round" in r.getMessage()]
            assert len(rounds) == 1 and "n=12" in rounds[0] and "transient_mib=400" in rounds[0]
            assert vpw._STATE["rounds"] == 0 and cuda.peak == 30000      # re-based at the extend's start
            cuda.peak = 30380                                             # the extend's own peak
            extend_trim.measure_close(runner, _fb(1100), cuda, rank=0)
            line = vpw.on_forward_end(types.SimpleNamespace(tp_rank=0), types.SimpleNamespace(
                forward_mode=types.SimpleNamespace(is_extend=lambda: True, is_decode=lambda: False,
                                                   is_target_verify=lambda: False),
                input_ids=types.SimpleNamespace(shape=(1100,)), batch_size=1), cuda)
            extend_trim.lend_cost_settle(0, cuda)
        assert extend_trim.lend_cost_mib(1100) == 380.0
        assert line is not None and "phase=chunk rows=1100" in line and "transient_mib=380" in line
    finally:
        vpw._STATE.clear()
        vpw._STATE.update(saved)


def test_a1_without_open_rounds_the_open_folds_and_rebases_and_keeps_the_cumulative_peak():
    vpw = _vpw_state()
    saved = dict(vpw._STATE)
    try:
        with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override(FLOORS), envs.FLLIPER_PDFLIP_VRAM_PEAK.override(False):
            vpw._STATE.update(rounds=0, shadow_alloc=0)
            runner = types.SimpleNamespace(tp_rank=0)
            worker = types.SimpleNamespace(tp_rank=0, model_runner=runner)
            cuda = ResettingCuda(peak=30400, alloc=30000, reserved=31000)
            extend_trim.measure_open(worker, _fb(5), cuda)
            assert cuda.peak == 30000 and vpw._STATE["shadow_alloc"] == 30400 * MIB   # "since the pools" kept
            extend_trim.measure_close(runner, _fb(5), cuda, rank=0)       # the extend allocated nothing above
            extend_trim.lend_cost_settle(0, cuda)
        assert extend_trim.lend_cost_mib(5) == 0.0                        # a READ 0, not a masked one
    finally:
        vpw._STATE.clear()
        vpw._STATE.update(saved)


def test_a1_the_q694b_rate_still_compares_against_the_raw_peak():
    """The re-base must not change Q-694b: its transient counts only when the extend beat the RAW inherited peak."""
    vpw = _vpw_state()
    saved = dict(vpw._STATE)
    try:
        with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override(FLOORS), envs.FLLIPER_PDFLIP_VRAM_PEAK.override(False):
            vpw._STATE.update(rounds=0)
            runner = types.SimpleNamespace(tp_rank=1)
            worker = types.SimpleNamespace(tp_rank=1, model_runner=runner)
            cuda = ResettingCuda(peak=32000, alloc=30000, reserved=31000)    # raw inherited peak 2000 over
            extend_trim.measure_open(worker, _fb(4096), cuda)
            cuda.peak, cuda.reserved = 31500, 31000 + 1024                   # extend 1500 < raw 2000
            line = extend_trim.measure_close(runner, _fb(4096), cuda, rank=1)
            assert line is not None and "transient_mib=0 reserved_growth_mib=1024" in line   # as before the fix
            extend_trim.lend_cost_settle(1, cuda)
            assert extend_trim.lend_cost_mib(4096) == 1500.0                 # LEND-COST reads the extend's own
    finally:
        vpw._STATE.clear()
        vpw._STATE.update(saved)


def test_round_start_closes_the_tail_before_any_lend_move():
    calls = []
    tms = h.CoreTms()
    with _Ctx(tms, 780):
        r, ctl = _tp0_five_rows(tms)
        extend_trim._CACHE["lend_cost_tail"] = (5, 449 * MIB, 0, 31000 * MIB)
        with mock.patch.object(extend_trim, "lend_cost_settle",
                               side_effect=lambda *a, **k: calls.append(("settle", k.get("lend_open")))), \
                mock.patch.object(dtl, "poll", side_effect=lambda *a, **k: calls.append(("poll", None))), \
                mock.patch.object(dtl, "_keep_short", side_effect=lambda *a, **k: calls.append(("move", None))):
            dtl.round_start(r.sched)
    assert calls[0] == ("settle", True) and calls[1][0] == "poll" and calls[-1][0] == "move"


# ---- (5) armed where a lend floor is, the rate path untouched ------------------------------------------

def test_tp0_opens_the_reading_without_q694b_and_its_rate_stays_unmeasured():
    with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override(FLOORS):
        assert not extend_trim.measure_armed(0) and extend_trim.lend_cost_armed(0)
        runner = types.SimpleNamespace(tp_rank=0)
        worker = types.SimpleNamespace(tp_rank=0, model_runner=runner)
        batch = types.SimpleNamespace(forward_mode=_Mode())
        with mock.patch.object(extend_trim, "_before_extend_trim", lambda w, b: None), \
                mock.patch.object(extend_trim, "measure_open") as op:
            extend_trim.before_extend(worker, batch)
            assert op.called
        cuda = Cuda(peak=30000, alloc=30000, reserved=31000)
        extend_trim.measure_open(worker, _fb(4096), cuda)
        cuda.peak, cuda.reserved = 32000, 32000
        assert extend_trim.measure_close(runner, _fb(4096), cuda, rank=0) is None   # no EXTEND-RATE line
        assert extend_trim.measured_rate() is None and not dtl.rate_measured(0)


def test_a_rank_without_a_lend_floor_reads_nothing():
    """P group / no D lend: no floor env -> no reading, no tail (the old measure_close)."""
    with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override(""):
        assert not extend_trim.lend_cost_armed(0)
        worker = types.SimpleNamespace(tp_rank=0, model_runner=types.SimpleNamespace(tp_rank=0))
        with mock.patch.object(extend_trim, "_before_extend_trim", lambda w, b: None), \
                mock.patch.object(extend_trim, "measure_open") as op:
            extend_trim.before_extend(worker, types.SimpleNamespace(forward_mode=_Mode()))
            assert not op.called
        extend_trim._CACHE["pending"] = (None, 0, 0, 0)
        assert extend_trim.measure_close(object(), _fb(5), Cuda(1, 1, 1), rank=0) is None
        assert extend_trim._CACHE["pending"] is None and extend_trim._CACHE["lend_cost_tail"] is None


def test_the_q694b_rate_line_of_a_measuring_rank_is_unchanged():
    with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override(FLOORS), \
            envs.FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB.override(RATE):
        runner = types.SimpleNamespace(tp_rank=1)
        worker = types.SimpleNamespace(tp_rank=1, model_runner=runner)
        cuda = Cuda(peak=1000, alloc=900, reserved=5000)
        extend_trim.measure_open(worker, _fb(4096), cuda)
        cuda.peak, cuda.reserved = 2200, 7200
        line = extend_trim.measure_close(runner, _fb(4096), cuda, rank=1)
        assert line is not None and "rate=0.5372" in line and extend_trim.measured_rate() == 0.5372
        # and the same reading fed the class record (4096 rows -> class 4096)
        extend_trim.lend_cost_settle(1, cuda)
        assert extend_trim.lend_cost_mib(4096) == 2200.0


# ---- (6) the HALT instrument ---------------------------------------------------------------------------

def test_an_extend_batch_ending_under_the_halt_line_is_named(caplog):
    with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override(FLOORS), caplog.at_level(logging.WARNING):
        runner = types.SimpleNamespace(tp_rank=0)
        worker = types.SimpleNamespace(tp_rank=0, model_runner=runner)
        cuda = Cuda(peak=30000, alloc=30000, reserved=31000)
        extend_trim.measure_open(worker, _fb(5), cuda)
        cuda.peak = 30449
        extend_trim.measure_close(runner, _fb(5), cuda, rank=0)
        cuda.free_mib = 299
        line = extend_trim.lend_cost_settle(0, cuda, lend_open=True, rows_on=40)
    assert line.startswith("EXTEND-LEND-COST HALT rank=0 rows=5 class=64 card_free_mib=299")
    assert "lend_open=1 rows_on=40" in line
    assert any("EXTEND-LEND-COST HALT" in r.getMessage() for r in caplog.records)
    with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override(FLOORS):
        assert "HALT" not in (_extend(5, transient=10, free_end=300) or "")


# ---- (7) switch off ------------------------------------------------------------------------------------

def test_switch_off_is_the_old_price_and_no_reading():
    with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override(FLOORS), \
            envs.FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB.override(RATE), \
            envs.FLLIPER_PDFLIP_D_LEND_MEASURED_PRICE.override(False):
        assert not extend_trim.lend_cost_armed(0)
        assert _extend(5, transient=449) is None
        assert extend_trim._CACHE["lend_cost"] is None
        extend_trim._CACHE["lend_cost"] = {64: (449, 64)}                 # even a held value is not read
        assert dtl.short_extend_need_mib(0, 5) == pytest.approx(SUM_5)


def test_the_switch_defaults_on():
    assert envs.FLLIPER_PDFLIP_D_LEND_MEASURED_PRICE.get() is True


# ---- (8) a measured-rate rank: the term only raises -----------------------------------------------------

def test_on_a_measured_rate_rank_the_measured_term_only_raises_the_max():
    with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override(FLOORS), \
            envs.FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB.override(RATE), \
            mock.patch.object(dtl, "rate_measured", lambda rank: True):
        assert dtl.short_extend_need_mib(1, 4) == pytest.approx(700.0)
        _extend(4, transient=20, rank=1)                                  # rc28 TP1: <= 20 MiB at <= 64 rows
        assert dtl.short_extend_need_mib(1, 4) == pytest.approx(700.0)    # unchanged
        _extend(4, transient=450, rank=1)
        assert dtl.short_extend_need_mib(1, 4) == pytest.approx(750.0)    # the measurement wins


def test_the_stage_keep_price_on_tp0_follows_the_measurement():
    ctl = types.SimpleNamespace(lent_from=16)
    sched = types.SimpleNamespace(waiting_queue=[object()], chunked_req=None, ps=types.SimpleNamespace(tp_rank=0))
    with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override(FLOORS), \
            envs.FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB.override(RATE), \
            envs.FLLIPER_PDFLIP_D_LEND_KEEP_SHORT.override(True), \
            mock.patch.object(dtl, "short_pending_rows", lambda s: 5):
        assert dtl.stage_keep_need_bytes(sched, ctl) == int(SUM_5 * MIB)
        _extend(5, transient=449)
        assert dtl.stage_keep_need_bytes(sched, ctl) == 767 * MIB


# ---- (9) rank-unity: no verdict reads the measured cost -------------------------------------------------

def test_the_verdicts_never_read_the_measured_cost():
    s = types.SimpleNamespace(chunked_req=None, waiting_queue=[s78._short(priced=1)],
                              running_batch=types.SimpleNamespace(reqs=[1, 2, 3, 4]))
    with mock.patch.object(extend_trim, "lend_cost_mib", side_effect=AssertionError("verdict read the cost")), \
            mock.patch.object(dtl, "measured_cost_mib", side_effect=AssertionError("verdict read the cost")):
        assert dtl.short_pending_rows(s) == 5
        gate = dtl.LendGate(2)
        assert [gate.step(k) for k in ("decode", "decode", "extend")] == ["hold", "lend", "return"]


# EXTEND-KORRIDOR (NF rc33 22.10.): the width vote, the short-extend price and the HALT/after-extend trim keep the card's
# corridor law (extend_trim.corridor_line_mib); this module pins the arithmetic it was written for at the old 300 MiB
# halt line -- the law cases are pinned in test_nf_rc33_extendkorridor_1022.
import pytest as _pytest_line300  # noqa: E402


@_pytest_line300.fixture(autouse=True)
def _extend_line_300_for_this_module(monkeypatch):
    from flliper.srt.pdflip import extend_trim as _et

    monkeypatch.setenv("FLLIPER_CORRIDOR_LAW_FLOOR_MIB", "300")
    _et._CACHE["line"] = getattr(_et, "_UNSET", None)
    yield
    _et._CACHE["line"] = getattr(_et, "_UNSET", None)
