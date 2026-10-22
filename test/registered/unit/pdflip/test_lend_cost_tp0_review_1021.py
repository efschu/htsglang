"""Review LEND-COST (nf-rc30-tp0lend-1021-review, 21.10.): the gaps of 9b6855e445 / c099fac60d as desk tests.

Base: ``test_lend_cost_tp0_1021.py`` (the author's 24) -- its fakes are reused, nothing re-implemented.

Metal (NF rc30 D ``boot_weg2_..._1d822bd150_1021_063252.D.log``, counted by the reviewer, not estimated):

* TP0 PDFLIP-VRAM-PEAK: 2202 of 2893 ``phase=chunk`` records have ``window_ms < 50`` and ``transient_mib=0`` --
  the extend's own forward end closes the open ``phase=round`` window FIRST (``vram_peak_window.on_forward_end``),
  so the extend's peak lands in the round line and the chunk line holds 0..5 ms after it. The LEND-COST open
  (``measure_open``) therefore runs on a peak counter that still holds the decode rounds since the last round
  record: TP0 pure-decode round windows transient p50 48 / p90 261 / p99 349 / max 411 MiB (2224 windows).
* An extend whose own peak stays under that decode peak is read with ``transient = 0`` (pa1 == pa0) -- the
  author pins that on purpose (``test_the_tail_closes_once_and_a_peak_the_extend_did_not_raise_is_not_its_cost``).
  Its true transient is then UNKNOWN in [0, pa0 - a0]; recording 0 (or the reserved growth alone, which the
  allocator cache can absorb -- 953 MiB cache left after the trim at 12:05:03) as the class cost is a lower
  bound sold as a measurement (finding A1; fixed by the author 21.10., the test below is green, no marker).
* On THIS metal the hole is covered by TP0's floor: a masked transient is <= 411 MiB, the floor prices
  ``300 + cost`` up to 467 MiB -> the TP0 price stays 767 (56 MiB margin). On a measured-rate rank (TP1/TP2)
  the term only raises, so a masked reading is the KEEP-PRICE baseline, no new risk.
* round_start prices the PENDING work (front-priced uncached, 0..5 rows on the metal) but the built extend is
  wider: of the round_start short returns on TP0 with a <= 64-row pending price, 4 built a <= 64-row extend,
  580 a wider one (128: 26, 256: 272, 512: 105, 1024: 159, 2048: 15, 4096: 3). ``on_batch`` reprices the
  BUILT rows before the extend runs -- that is the real guard; pinned below.
"""
from __future__ import annotations

import importlib.util
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import d_transient_lend as dtl  # noqa: E402
from flliper.srt.pdflip import extend_trim  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_t_lend_cost_author_1021", os.path.join(os.path.dirname(__file__), "test_lend_cost_tp0_1021.py"))
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

MIB = A.MIB
FLOORS, RATE = A.FLOORS, A.RATE
#: TP0 pure-decode PDFLIP-VRAM-PEAK round windows, rc30 D 1021_063252 (2224 windows): the largest decode peak
#: an extend's open can inherit from the counter
TP0_DECODE_TRANSIENT_MAX_MIB = 411
TP0_FLOOR_MIB = 767


@pytest.fixture(autouse=True)
def _fresh():
    extend_trim.reset_for_tests()
    with envs.FLLIPER_PDFLIP_EXTEND_RATE_MEASURE.override(True), \
            envs.FLLIPER_PDFLIP_EXTEND_RATE_MEASURE_RANKS.override("1,2"):
        yield
    extend_trim.reset_for_tests()


def _masked_extend(rows, decode_peak_mib, true_transient_mib, growth_mib=0, cache_mib=953):
    """One extend opened while the counter still holds a decode peak ``decode_peak_mib`` above the start
    allocation; the extend's own transient stays under it and the allocator cache absorbs it."""
    runner = types.SimpleNamespace(tp_rank=0)
    worker = types.SimpleNamespace(tp_rank=0, model_runner=runner)
    cuda = A.Cuda(peak=30000 + decode_peak_mib, alloc=30000, reserved=30000 + cache_mib)
    extend_trim.measure_open(worker, A._fb(rows), cuda)
    assert true_transient_mib <= decode_peak_mib            # the extend did not raise the counter
    cuda.reserved += growth_mib
    extend_trim.measure_close(runner, A._fb(rows), cuda, rank=0)
    return extend_trim.lend_cost_settle(0, cuda)


# ---- A1: a masked transient is recorded as a measured 0 ---------------------------------------------------

def test_a1_a_masked_transient_is_never_recorded_below_its_bound():
    with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override(FLOORS):
        _masked_extend(1100, decode_peak_mib=400, true_transient_mib=380)
        cost = extend_trim.lend_cost_mib(1100)
    assert cost is None or cost >= 400


def test_a1_the_masked_reading_books_its_bound_and_prices_a_2048_class_extend_by_it():
    """After the A1 fix (author 21.10.): the masked class-2048 reading books the inherited peak (400 MiB) as its
    bound -- an upper bound of what that extend cost (true 380) -- and the 1100-row price is
    ``max(767, 300 + 0.5871 x 1100 = 946, 300 + 400)``; before the fix it booked 0. (Review doc test, re-pinned.)"""
    with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override(FLOORS), \
            envs.FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB.override(RATE):
        assert dtl.short_extend_need_mib(0, 1100) == pytest.approx(767 + 300 + 0.5871 * 1100)
        _masked_extend(1100, decode_peak_mib=400, true_transient_mib=380)
        assert extend_trim.lend_cost_mib(1100) == 400.0
        assert dtl.short_extend_need_mib(0, 1100) == pytest.approx(max(767, 300 + 0.5871 * 1100, 700))


def test_a1_on_the_metal_the_tp0_floor_covers_every_maskable_transient():
    """The arithmetic that makes A1 latent on rc30, not live: the largest decode peak an open can inherit
    (411 MiB) is under what TP0's floor already prices (floor - 300 = 467) -> a masked <= 64 / 128..512 class
    reading cannot lower TP0's price under 767. A config with a smaller floor or larger decode transients
    loses that cover -- A1 must not depend on it."""
    assert TP0_DECODE_TRANSIENT_MAX_MIB <= TP0_FLOOR_MIB - extend_trim.CHUNKING_FLOOR_MIB
    with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override(FLOORS), \
            envs.FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB.override(RATE):
        _masked_extend(5, decode_peak_mib=TP0_DECODE_TRANSIENT_MAX_MIB, true_transient_mib=TP0_DECODE_TRANSIENT_MAX_MIB)
        assert dtl.short_extend_need_mib(0, 5) == pytest.approx(767.0)
        # and even the bound itself as the cost keeps the floor
        assert max(767.0, 300.0 + TP0_DECODE_TRANSIENT_MAX_MIB) == 767.0


# ---- the built extend is the guard, not the pending price -------------------------------------------------

def _built(rows):
    return types.SimpleNamespace(forward_mode=types.SimpleNamespace(
        is_decode=lambda: False, is_target_verify=lambda: False, is_idle=lambda: False,
        is_extend=lambda: True), extend_num_tokens=rows)


def test_round_start_keeps_on_the_pending_class_and_on_batch_returns_for_the_wider_built_extend():
    """Metal: pending rows=5 (front-priced 1 + 4 decodes), built 1285. round_start keeps at the <= 64 price
    (767); the built 1285-row extend has no class-2048 reading -> on_batch prices the old sum and gives rows
    back BEFORE the extend allocates."""
    with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override(FLOORS):
        A._extend(5, transient=449)
    out, ctl, _ = A._round_start_at(780)
    assert out == "keep" and ctl.rows_on == 40
    tms = A.h.CoreTms()
    with A._Ctx(tms, 780):
        r, ctl = A._tp0_five_rows(tms)
        assert dtl.on_batch(r.sched, _built(1285)) == "return" and ctl.rows_on < 40


def test_on_batch_with_a_measured_but_narrower_class_reading_keeps_the_old_sum():
    """Class 2048 measured up to 1100 rows only: a built 1285-row extend is wider than the widest reading ->
    the sum (767 + 300 + 0.5871 x 1285 = 1821), not ``300 + cost``."""
    with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override(FLOORS), \
            envs.FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB.override(RATE):
        A._extend(1100, transient=900)
        assert dtl.short_extend_need_mib(0, 1285) == pytest.approx(767 + 300 + 0.5871 * 1285)
    tms = A.h.CoreTms()
    with A._Ctx(tms, 1300):
        r, ctl = A._tp0_five_rows(tms)
        assert dtl.on_batch(r.sched, _built(1285)) == "return" and ctl.rows_on < 40


# ---- the reading after a flip leg / sleep -----------------------------------------------------------------

def test_a_reserved_drop_between_the_close_and_the_settle_never_lowers_the_cost():
    """A leg (sleep / flip release) between the target close and the next pass empties the cache: the draft
    term is clipped at 0, the forward reading stands, nothing goes negative."""
    with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override(FLOORS):
        line = A._extend(300, transient=250, growth=120, draft=-900)
    assert "cost_mib=250" in line and "draft_growth_mib=0" in line
    assert extend_trim.lend_cost_mib(300) == 250.0


def test_the_reading_runs_with_q694b_switched_off_entirely():
    """TP0's reading does not lean on FLLIPER_PDFLIP_EXTEND_RATE_MEASURE: with it OFF the D lend floor alone
    arms the open/close/settle."""
    with envs.FLLIPER_PDFLIP_EXTEND_RATE_MEASURE.override(False), \
            envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override(FLOORS):
        extend_trim.reset_for_tests()
        assert not extend_trim.measure_armed(0) and extend_trim.lend_cost_armed(0)
        line = A._extend(5, transient=120)
    assert "EXTEND-LEND-COST rank=0 rows=5 class=64 cost_mib=120" in line
    assert extend_trim.measured_rate() is None


def test_a_second_target_extend_before_a_settle_loses_a_reading_never_lowers_one():
    """Two target closes without a scheduler pass between: the first tail is replaced (a lost reading), the
    class maximum of earlier readings stands."""
    with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override(FLOORS):
        A._extend(5, transient=300)
        runner = types.SimpleNamespace(tp_rank=0)
        worker = types.SimpleNamespace(tp_rank=0, model_runner=runner)
        for t in (440, 10):
            cuda = A.Cuda(peak=30000, alloc=30000, reserved=31000)
            extend_trim.measure_open(worker, A._fb(5), cuda)
            cuda.peak = 30000 + t
            extend_trim.measure_close(runner, A._fb(5), cuda, rank=0)
        extend_trim.lend_cost_settle(0, cuda)
    assert extend_trim.lend_cost_mib(5) == 300.0


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
