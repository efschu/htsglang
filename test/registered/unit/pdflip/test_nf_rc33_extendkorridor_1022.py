"""EXTEND-KORRIDOR (NF rc33 22.10., desk/nf-rc33-extendkorridor-1022): the D extend keeps the corridor law.

Metal (NF rc33 D ``boot_weg2_..._fbb680d6d3_1022_060647.D.log``, TP0 = RTX 5090, law 767 MiB):

    06:17:04 PDFLIP-EXTEND-CACHE-TRIM rank=0 released=216 card_free_before=1855 card_free_after=2071
    06:17:04 PDFLIP-EXTEND-STUECKELUNG rank=0 geplant=4096 cap=3008 post=2071 rate=0.5871 floor=300
    06:17:08 PDFLIP-VRAM-PEAK rank=0 phase=chunk rows=2710 ... transient_mib=1416 allocated_mib=30496
             reserved_mib=33116 card_free_start_mib=2071 card_free_mib=47
    06:17:08 EXTEND-LEND-COST rank=0 rows=2710 class=4096 cost_mib=2024 ... card_free_mib=46
    06:17:10 CORRIDOR LAW BREACHED ... 44 MiB, 723 MiB BELOW the 767 MiB law
    06:17:13 PDFLIP-VRAM-PEAK phase=round ... reserved_mib=33118 card_free_start_mib=47 card_free_mib=45
    06:17:14 PDFLIP-EXTEND-CACHE-TRIM rank=0 released=1970 card_free_before=45 card_free_after=2015

Earlier on TP0 the same boot: 06:16:13 HALT rows=2375 cost_mib=1788 (0.7528 MiB/row), 06:14:16 rows=2880 cost 1930.

What must hold:
(1) REPRO: the 06:17:04 state (post 2071, start rate 0.5871, law 767, TP0's own wide-extend cost 0.7529 MiB/row
    measured at 06:16:13) votes a chunk whose MEASURED cost (0.747 MiB/row, the 2710-row extend) leaves >= 767;
    HEAD voted 3008 (2710 rows ran whole, card 47).
(2) the vote's line is the corridor law from the guard's one rank-local door (its verdict floor), never 300;
    the cache is not counted as funds (the 27B rc29 double count never enters this line).
(3) DANGER (mutant "extend under the corridor allowed"): over a grid of card states, every vote above one page
    leaves ``post - cap x rate >= line``.
(4) the LEND-COST reading ratchets the wide-extend per-row cost (rows >= 2048 only, no factor) into the rate.
(5) after an extend batch that ends under the line the cache goes back to the driver at the next pass (trim),
    and the HALT instrument names the corridor line (not 300).
(6) the short-extend price uses the same line: 900 rows kept at 1042 on 1094 free (card ended 471) now return.
"""
from __future__ import annotations

import os
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.managers import corridor_guard as CG  # noqa: E402
from flliper.srt.pdflip import d_transient_lend as dtl  # noqa: E402
from flliper.srt.pdflip import extend_trim as ET  # noqa: E402

MIB = 1 << 20
PAGE = 64
START = 0.5871
LAW = 767.0
FLOORS = "767,700,701"
#: the 2710-row extend's measured cost per row (reserved growth 2024 MiB)
METAL_RATE_2710 = 2024 / 2710
#: the real line reader (the autouse fixture patches the module attribute)
_ORIG_READ = getattr(ET, "_read_corridor_line", None)  # HEAD (fbb680d6d3) has no reader: its tests go red, not uncollectable


@pytest.fixture(autouse=True)
def _fresh():
    ET.reset_for_tests()
    with envs.FLLIPER_PDFLIP_EXTEND_RATE_MEASURE.override(True), \
            envs.FLLIPER_PDFLIP_EXTEND_RATE_MEASURE_RANKS.override("1,2"), \
            envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override(FLOORS), \
            envs.FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB.override("0.5871,0.5871,0.5871"), \
            mock.patch.object(ET, "_read_corridor_line", lambda: (LAW, "MEASURED-D:767"), create=True):
        yield
    ET.reset_for_tests()


class Cuda:
    """One card: driver free, torch allocated / reserved / peak (MiB); empty_cache returns the free cache."""

    def __init__(self, free, alloc=30426.0, reserved=31092.0, peak=None, releasable=None):
        self.free, self.alloc, self.reserved = float(free), float(alloc), float(reserved)
        self.peak = float(alloc if peak is None else peak)
        self.releasable = (self.reserved - self.alloc) if releasable is None else float(releasable)
        self.trims = 0

    def is_current_stream_capturing(self):
        return False

    def mem_get_info(self):
        return int(self.free * MIB), int(32088 * MIB)

    def memory_reserved(self):
        return int(self.reserved * MIB)

    def memory_allocated(self):
        return int(self.alloc * MIB)

    def memory_stats(self):
        return {"allocated_bytes.all.peak": int(self.peak * MIB),
                "allocated_bytes.all.current": int(self.alloc * MIB),
                "reserved_bytes.all.current": int(self.reserved * MIB)}

    def synchronize(self):
        pass

    def empty_cache(self):
        self.trims += 1
        self.free += self.releasable
        self.reserved -= self.releasable
        self.releasable = 0.0


class _Mode:
    def is_extend(self):
        return True

    def is_target_verify(self):
        return False


def _fb(rows):
    return types.SimpleNamespace(forward_mode=_Mode(), input_ids=types.SimpleNamespace(shape=(rows,)))


def _extend(rows, growth, rank=0, free_end=2000.0, releasable=None):
    """One D extend batch on ``rank``: open, target close (reserved grows by ``growth``), next-pass settle."""
    runner = types.SimpleNamespace(tp_rank=rank)
    worker = types.SimpleNamespace(tp_rank=rank, model_runner=runner)
    cuda = Cuda(free=free_end + growth, alloc=30426.0, reserved=31092.0)
    ET.measure_open(worker, _fb(rows), cuda)
    cuda.reserved += growth
    cuda.peak = cuda.alloc + growth * 0.7
    cuda.alloc += 70.0
    cuda.free = free_end
    cuda.releasable = (growth if releasable is None else releasable)
    ET.measure_close(runner, _fb(rows), cuda, rank=rank)
    line = ET.lend_cost_settle(rank, cuda)
    return line, cuda


# ---- (1) the repro ------------------------------------------------------------------------------------

def test_repro_0617_tp0_vote_leaves_the_law_at_the_measured_cost():
    # 06:16:13: TP0's own 2375-row extend cost 1788 MiB (the HALT line of that boot) -- the rank knows its cost
    _extend(2375, 1788, rank=0, free_end=284.0)
    assert ET.lend_rate(0) == pytest.approx(0.7529)
    cuda = Cuda(free=2071.0)
    cap = ET.width_vote(cuda, 0, 4096, PAGE, True)
    assert cap is not None and cap < 2710                       # HEAD: 3008 -> the 2710 rows ran whole
    assert cap == 1728                                         # floor((2071 - 767) / 0.7529) = 1732 -> page
    assert 2071.0 - cap * METAL_RATE_2710 >= LAW               # the metal cost per row leaves the law
    assert 2071.0 - 2710 * METAL_RATE_2710 < 300               # what ran: card 47


def test_line_alone_already_narrows_the_start_rate_vote():
    cap = ET.width_vote(Cuda(free=2071.0), 0, 4096, PAGE, True)
    assert cap == 2176                                         # floor((2071 - 767) / 0.5871) = 2221 -> page
    assert 2071.0 - cap * START >= LAW


# ---- (2) the line: the guard's verdict floor, never 300; the cache funds nothing ----------------------

def test_the_line_is_read_through_the_guards_rank_local_door():
    measured = CG.CorridorFloor(card_uuid="u", group="D", transient_mib=767, reserve_mib=0,
                                source=CG.measured_floor_source("D"))
    calls = []

    def door(*a, **k):
        calls.append(1)
        return measured

    assert _ORIG_READ is not None
    with mock.patch.object(CG, "corridor_floor_for_current_device", door):
        mib, src = _ORIG_READ()
    assert mib == 767.0 and src.endswith(":767") and calls == [1]


@pytest.mark.parametrize("transient,source,want", [
    (767, "MEASURED", 767.0),           # measured: no band under it
    (1024, CG.FLOOR_SOURCE_FALLBACK, 819.0),   # stated fallback: the -20 % band floor, as the guard grades it
    (100, "MEASURED", 300.0),           # never under the HALT line
])
def test_corridor_line_grades_like_the_guard(transient, source, want):
    src = CG.measured_floor_source("D") if source == "MEASURED" else source
    fl = CG.CorridorFloor(card_uuid="u", group="D", transient_mib=transient, reserve_mib=0, source=src)
    ET.reset_for_tests()
    assert _ORIG_READ is not None
    with mock.patch.object(ET, "_read_corridor_line", _ORIG_READ), \
            mock.patch.object(CG, "corridor_floor_for_current_device", lambda *a, **k: fl):
        assert ET.corridor_line_mib(0) == pytest.approx(want)
    # read once per process: a later door answer does not move it
    with mock.patch.object(CG, "corridor_floor_for_current_device", lambda *a, **k: None):
        assert ET.corridor_line_mib(0) == pytest.approx(want)


def test_torch_cache_does_not_fund_the_chunk():
    lean = ET.width_vote(Cuda(free=2071.0, alloc=30426.0, reserved=31092.0), 0, 4096, PAGE, True)
    ET._CACHE["logged_cap"] = None
    fat = ET.width_vote(Cuda(free=2071.0, alloc=30426.0, reserved=33092.0), 0, 4096, PAGE, True)
    assert lean == fat == 2176


# ---- (3) danger: no vote lets an extend end under the line --------------------------------------------

@pytest.mark.parametrize("lend", [None, 0.6, 0.7529, 0.9])
@pytest.mark.parametrize("post", [800, 1000, 1500, 2071, 2500, 3000, 3500, 4000])
def test_danger_no_vote_funds_an_extend_under_the_corridor(post, lend):
    ET.reset_for_tests()
    if lend is not None:
        ET._CACHE["lend_rate"] = lend
    cap = ET.width_vote(Cuda(free=float(post)), 0, 4096, PAGE, True)
    rate = max(START, lend or 0.0)
    width = 4096 if cap is None else cap
    if width > PAGE:                                           # one page is the progress floor, not a funding
        assert post - width * rate >= LAW - 1e-6, (post, lend, width)


# ---- (4) the wide-extend per-row cost enters the rate -------------------------------------------------

def test_lend_rate_ratchets_on_wide_extends_only():
    _extend(900, 742, free_end=370.0)                          # narrow: fixed part, not a per-row rate
    assert ET.lend_rate(0) is None
    _extend(2880, 1930, free_end=206.0)
    assert ET.lend_rate(0) == pytest.approx(0.6702)
    _extend(2375, 1788, free_end=284.0)
    assert ET.lend_rate(0) == pytest.approx(0.7529)
    _extend(2752, 1852, free_end=218.0)                        # 0.673: under the maximum, no move
    assert ET.lend_rate(0) == pytest.approx(0.7529)
    assert ET.effective_rate(START, 0) == pytest.approx(0.7529)


def test_lend_rate_needs_lend_cost_armed():
    # mock, not envs.override: an override inside a failing test body never restores (generator without finally)
    ET._CACHE["lend_rate"] = 0.75
    with mock.patch.object(ET, "lend_cost_armed", lambda rank=None: False):
        assert ET.lend_rate(0) is None
        assert ET.effective_rate(START, 0) == pytest.approx(START)


# ---- (5) after the extend: the cache goes back, HALT names the corridor line -------------------------

def test_after_extend_under_the_line_the_cache_goes_back_at_the_next_pass():
    line, cuda = _extend(2710, 2024, free_end=46.0, releasable=1970.0)
    assert cuda.trims == 1 and cuda.free == pytest.approx(2016.0)
    assert "EXTEND-LEND-COST HALT rank=0 rows=2710" in line
    assert "under the 767 MiB line" in line and "trim_after_mib=2016" in line


def test_between_halt_and_law_is_a_breach_too():
    line, cuda = _extend(2024, 1248, free_end=500.0, releasable=600.0)
    assert cuda.trims == 1                                     # HEAD: 500 >= 300 -> no HALT, no trim
    assert "HALT" in line and "767 MiB line" in line


def test_over_the_line_no_trim_no_halt():
    line, cuda = _extend(2024, 1248, free_end=900.0, releasable=600.0)
    assert cuda.trims == 0
    assert line is None or "HALT" not in line


# ---- (6) the short-extend price keeps the same line ----------------------------------------------------

def test_short_price_0617_02_900_rows_returns_instead_of_keeping():
    _extend(901, 742, free_end=370.0)                          # class 1024 measured at 742 MiB (06:16:27)
    need = dtl.short_extend_need_mib(0, 900)
    assert need == pytest.approx(LAW + 742.0)                  # HEAD: 300 + 742 = 1042 -> kept on 1094
    assert need > 1094.0


def test_short_price_tiny_extend_without_reading_stays_the_floor_sum():
    # TP0 start-rate rank, no class reading: review A's sum floor + 300 + rate x rows, never under line + rate x rows
    need = dtl.short_extend_need_mib(0, 5)
    assert need == pytest.approx(max(767 + 300 + START * 5, LAW + START * 5))
