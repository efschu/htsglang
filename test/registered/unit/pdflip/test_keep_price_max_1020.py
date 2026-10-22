"""KEEP-PRICE (20.10., nf-expert-h2d-1020): a short D extend keeps the lent expert rows up to
``max(lend floor, 300 + rate x rows)`` free MiB -- no longer ``floor + 300 + rate x rows``.

Metal (rc28 D ``..._1020_174205``, 2.4 h, all three ranks): 2184 KEEP-SHORT returns before short extends
(TP1 732, 75.2 s return host + fence waits), every one priced ``need_mib = floor + 300 + rate x rows``
(TP1 4-row extend: 1002 MiB) while the extends' card use was <= 20 MiB (<= 64 rows) / <= 288 MiB
(<= 512 rows) on TP1 (PDFLIP-VRAM-PEAK phase=chunk). At the new price 1795 of the 2184 returns need no
return at all; started at exactly the new price every measured extend (752 + 101 windows) keeps >= 397
MiB free at its peak.

What must hold (the S78 desk rank, lattice step 4 rows, S0 = 32 rows ON, lend to 40):
(1) ``short_extend_need_mib`` = max(floor, chunk price); None without floor or rate (= full return);
(2) floor under the chunk price: the chunk price alone decides (keep at exactly it, return 1 MiB under);
(3) floor above the chunk price: the floor alone decides (keep at exactly it, return 1 MiB under) --
    DANGER (OOM): a kept lend never leaves the card under the floor;
(4) danger (OOM): whatever stays after a return, free + released >= the price;
(5) the built extend's exact rows are priced the same way (``on_batch``);
(6) the S106 stage-keep price is the same max (floor alone with nothing pending);
(7) a big / unpriced pending extend still returns everything (the width is worth more than a row there);
(8) review A (20.10.): the max applies ONLY on a rank whose chunk rate is MEASURED (Q-694b armed for the
    rank AND a measurement held). A start-rate rank (NF TP0) or a rank before its first measurement keeps
    the S78 sum -- rc28 _201023 TP0 20:43:30, 2139 rows: card use 1596 MiB (trim-after before the forward
    to the extend's record) against a max price of 1556. Tests (1)-(7) run on a measured rank (fixture).
"""
from __future__ import annotations

import importlib.util
import os
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import d_transient_lend as dtl  # noqa: E402
from flliper.srt.pdflip import extend_trim  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_t_keep_short_for_price", os.path.join(os.path.dirname(__file__), "test_s78_d_lend_keep_short_1019.py"))
s78 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s78)
h = s78.h

MIB = 1 << 20
ROWS = s78.ROWS                      # 100 (95 priced + 5 running decodes)
CHUNK_MIB = 300 + 0.5 * ROWS         # rate 0.5 -> 350 MiB


@pytest.fixture(autouse=True)
def _fresh_trim_cache(request):
    extend_trim.reset_for_tests()
    if "unmeasured" in request.keywords or "gate" in request.node.name:
        yield
    else:
        # (1)-(7): a rank whose rate is measured (NF TP1/TP2) -- review A's gate open
        with mock.patch.object(dtl, "rate_measured", lambda rank: True):
            yield
    extend_trim.reset_for_tests()


# ---- (1) the price ----------------------------------------------------------------------------------

def test_the_price_is_the_larger_of_floor_and_chunk_price():
    with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override("767,700,701"), \
            envs.FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB.override("0.5871"):
        extend_trim.reset_for_tests()
        # the metal case: a 4-row extend on TP1 -- 1002 MiB before, the floor now
        assert dtl.short_extend_need_mib(1, 4) == pytest.approx(700.0)
        assert dtl.short_extend_need_mib(0, 4) == pytest.approx(767.0)
        # a 1413-row short extend (rc28 _201023 TP1, priced 1829 before): the chunk price alone
        assert dtl.short_extend_need_mib(1, 1413) == pytest.approx(300 + 0.5871 * 1413)
        # the crossing: under it the floor, over it the chunk price -- never the sum
        for rows in (0, 1, 680, 681, 3792):
            want = max(700.0, 300 + 0.5871 * rows)
            assert dtl.short_extend_need_mib(1, rows) == pytest.approx(want)
            assert dtl.short_extend_need_mib(1, rows) < 700.0 + 300 + 0.5871 * rows
    with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override(""), \
            envs.FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB.override("0.5871"):
        extend_trim.reset_for_tests()
        assert dtl.short_extend_need_mib(1, 4) is None          # no floor: unpriced
    with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override("700"), \
            envs.FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB.override(""):
        extend_trim.reset_for_tests()
        assert dtl.short_extend_need_mib(1, 4) is None          # no rate: unpriced


def test_the_measured_rate_still_raises_the_chunk_price():
    """The planner's run-time rate (Q-694b, ratchet x1.15) feeds the max unchanged."""
    with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override("700"), \
            envs.FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB.override("0.5871"), \
            mock.patch.object(extend_trim, "measure_armed", lambda rank=None: True):
        extend_trim.reset_for_tests()
        extend_trim._CACHE["measured"] = 0.6274
        eff = extend_trim.effective_rate(0.5871, 2)
        assert dtl.short_extend_need_mib(2, 2241) == pytest.approx(300 + eff * 2241)
        assert dtl.short_extend_need_mib(2, 10) == pytest.approx(700.0)


# ---- (2)/(3)/(4) round_start on the live desk rank ---------------------------------------------------

def _run(free_mib, floor_mib):
    tms = h.CoreTms()
    free = int(free_mib * MIB)
    with s78._Ctx(tms, free, floor=str(floor_mib)):
        r, ctl = s78._live_lent(tms)
        out = dtl.round_start(r.sched)
        released = ctl._bank_bytes_at(40) - ctl._bank_bytes_at(ctl.rows_on)
        return out, ctl, free, released


def test_floor_under_the_chunk_price_adds_nothing():
    out, ctl, _f, _r = _run(CHUNK_MIB, floor_mib=200)
    assert out == "keep" and ctl.rows_on == 40 and ctl.returned_rows == 0   # S78 returned here (550 MiB)
    out, ctl, free, released = _run(CHUNK_MIB - 1, floor_mib=200)
    assert out == "return" and ctl.rows_on < 40
    assert free + released >= CHUNK_MIB * MIB or ctl.lent_from is None     # (4)


def test_floor_above_the_chunk_price_is_the_price():
    floor = 600
    out, ctl, _f, _r = _run(floor, floor_mib=floor)
    assert out == "keep" and ctl.rows_on == 40                              # S78 returned here (950 MiB)
    # DANGER (OOM): one MiB under the floor never keeps the whole lend
    out, ctl, free, released = _run(floor - 1, floor_mib=floor)
    assert out == "return" and ctl.rows_on < 40
    assert free + released >= floor * MIB or ctl.lent_from is None         # (4)


@pytest.mark.parametrize("free_mib,floor_mib", [(330, 0), (340, 345), (560, 600), (100, 50), (20, 600)])
def test_danger_whatever_stays_the_card_holds_the_price(free_mib, floor_mib):
    out, ctl, free, released = _run(free_mib, floor_mib)
    price = max(floor_mib, CHUNK_MIB)
    assert out == "return"
    assert ctl.lent_from is None or free + released >= price * MIB


# ---- (5) the built extend -----------------------------------------------------------------------------

def test_the_built_extend_is_priced_by_the_same_max():
    tms = h.CoreTms()
    floor = 600
    with s78._Ctx(tms, floor * MIB, floor=str(floor)):
        r, ctl = s78._live_lent(tms)
        batch = types.SimpleNamespace(forward_mode=types.SimpleNamespace(
            is_decode=lambda: False, is_target_verify=lambda: False, is_idle=lambda: False,
            is_extend=lambda: True), extend_num_tokens=40)
        assert dtl.on_batch(r.sched, batch) == "keep" and ctl.rows_on == 40
    tms = h.CoreTms()
    with s78._Ctx(tms, (floor - 1) * MIB, floor=str(floor)):
        r, ctl = s78._live_lent(tms)
        batch = types.SimpleNamespace(forward_mode=types.SimpleNamespace(
            is_decode=lambda: False, is_target_verify=lambda: False, is_idle=lambda: False,
            is_extend=lambda: True), extend_num_tokens=40)
        assert dtl.on_batch(r.sched, batch) == "return" and ctl.rows_on < 40


# ---- (6) the S106 stage-keep price --------------------------------------------------------------------

def _price_sched(queue=()):
    return types.SimpleNamespace(waiting_queue=list(queue), chunked_req=None,
                                 ps=types.SimpleNamespace(tp_rank=1))


def test_the_stage_keep_price_is_the_same_max():
    ctl = types.SimpleNamespace(lent_from=16)
    with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override("767,700,701"), \
            envs.FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB.override("0.5871"), \
            envs.FLLIPER_PDFLIP_D_LEND_KEEP_SHORT.override(True):
        extend_trim.reset_for_tests()
        assert dtl.stage_keep_need_bytes(_price_sched(), ctl) == 700 * MIB
        with mock.patch.object(dtl, "short_pending_rows", lambda s: 5):
            # the rc28 _201023 cascade: keep_need_mib=1002 before, the floor now
            assert dtl.stage_keep_need_bytes(_price_sched([object()]), ctl) == 700 * MIB
        with mock.patch.object(dtl, "short_pending_rows", lambda s: 2000):
            assert dtl.stage_keep_need_bytes(_price_sched([object()]), ctl) == int((300 + 0.5871 * 2000) * MIB)


# ---- (7) big / unpriced: the full return as before ----------------------------------------------------

def test_a_big_pending_extend_still_returns_everything():
    tms = h.CoreTms()
    with s78._Ctx(tms, 10_000 * MIB, floor="100"):
        r, ctl = s78._live_lent(tms)
        r.sched.waiting_queue = [s78._short(priced=3200, halfx=3200)]   # not short by R2
        assert dtl.round_start(r.sched) == "return"
        assert ctl.rows_on == 32 and ctl.lent_from is None


# ---- (8) review A: the max only on a rank with a MEASURED rate ---------------------------------------

def test_gate_unmeasured_rank_keeps_the_s78_sum():
    """NF TP0 (start rate only): the sum, also for a tiny extend -- the max would leave the floor's fixed
    part uncovered (rc28 _201023 TP0 20:43:30)."""
    with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override("767,700,701"), \
            envs.FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB.override("0.5871"), \
            envs.FLLIPER_PDFLIP_EXTEND_RATE_MEASURE.override(True), \
            envs.FLLIPER_PDFLIP_EXTEND_RATE_MEASURE_RANKS.override("1,2"):
        extend_trim.reset_for_tests()
        extend_trim._CACHE["measured"] = 0.5131  # even with a value held: rank 0 is not armed
        assert not dtl.rate_measured(0)
        assert dtl.short_extend_need_mib(0, 2139) == pytest.approx(767 + 300 + 0.5871 * 2139)
        assert dtl.short_extend_need_mib(0, 4) == pytest.approx(767 + 300 + 0.5871 * 4)


def test_gate_armed_rank_before_its_first_measurement_keeps_the_sum():
    with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override("767,700,701"), \
            envs.FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB.override("0.5871"), \
            envs.FLLIPER_PDFLIP_EXTEND_RATE_MEASURE.override(True), \
            envs.FLLIPER_PDFLIP_EXTEND_RATE_MEASURE_RANKS.override("1,2"):
        extend_trim.reset_for_tests()
        assert extend_trim.measured_rate() is None
        assert not dtl.rate_measured(1)
        assert dtl.short_extend_need_mib(1, 4) == pytest.approx(700 + 300 + 0.5871 * 4)
        # the first measurement (>= 2048 rows) opens the gate on this rank
        extend_trim._CACHE["measured"] = 0.5131
        assert dtl.rate_measured(1)
        eff = extend_trim.effective_rate(0.5871, 1)
        assert dtl.short_extend_need_mib(1, 4) == pytest.approx(700.0)
        assert dtl.short_extend_need_mib(1, 2000) == pytest.approx(300 + eff * 2000)


def test_gate_measurement_switched_off_keeps_the_sum():
    with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override("700"), \
            envs.FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB.override("0.5871"), \
            envs.FLLIPER_PDFLIP_EXTEND_RATE_MEASURE.override(False):
        extend_trim.reset_for_tests()
        extend_trim._CACHE["measured"] = 0.5131
        assert not dtl.rate_measured(1)
        assert dtl.short_extend_need_mib(1, 4) == pytest.approx(700 + 300 + 0.5871 * 4)


def test_gate_danger_unmeasured_live_rank_returns_at_the_max_price():
    """DANGER (OOM) on the live desk rank: an unmeasured rank with free = max price (but under the sum)
    gives rows back -- the max must not apply there."""
    tms = h.CoreTms()
    floor = 200
    with s78._Ctx(tms, int(CHUNK_MIB * MIB), floor=str(floor)), \
            envs.FLLIPER_PDFLIP_EXTEND_RATE_MEASURE.override(False):
        r, ctl = s78._live_lent(tms)
        assert dtl.round_start(r.sched) == "return" and ctl.rows_on < 40
        released = ctl._bank_bytes_at(40) - ctl._bank_bytes_at(ctl.rows_on)
        assert int(CHUNK_MIB * MIB) + released >= int((floor + CHUNK_MIB) * MIB) or ctl.lent_from is None


def test_gate_stage_keep_on_an_unmeasured_rank_is_the_sum():
    ctl = types.SimpleNamespace(lent_from=16)
    with envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override("767,700,701"), \
            envs.FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB.override("0.5871"), \
            envs.FLLIPER_PDFLIP_D_LEND_KEEP_SHORT.override(True), \
            envs.FLLIPER_PDFLIP_EXTEND_RATE_MEASURE.override(False):
        extend_trim.reset_for_tests()
        with mock.patch.object(dtl, "short_pending_rows", lambda s: 5):
            assert dtl.stage_keep_need_bytes(_price_sched([object()]), ctl) == int((700 + 300 + 0.5871 * 5) * MIB)


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
