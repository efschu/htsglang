# SPDX-License-Identifier: Apache-2.0
"""EXTEND-CAP-SOFT (27.10.): the D width vote funds down to the HARD floor, not to the 1024 MiB plan line.

Boot dkr27bbar1drei10272118_fdd27a84d4_1027_211839 (D group TP3 INT8, --chunked-prefill-size 16384), D TP2 (RTX 3080):
vote ``post`` 1062-1184 MiB, line 1024 (UNMEASURED-FALLBACK), rate 0.3091 (``rate_src=start``) -> cap 122-516 -> the Q-710
floor (16384 / 16 = 1024) -> ``GROUP-NARROWED 16384 -> 1024``. PDFLIP-VRAM-PEAK phase=chunk TP2, rows=1024: transient
139 MiB (0.136 MiB/row), card free 904 -> 846. The card ended every chunk ~450 MiB above the 400 MiB near-OOM edge.

User rule 24.10.: the ~1 GB line is a plan target, not a run-time gate; only near-OOM and the user's reserve are hard.
The numbers below are that boot's; the hard floor is near-OOM 400 + F 238 = 638 MiB (``SOFT_FIXED_DEFAULT_MIB``).
"""

import logging
import types
from contextlib import ExitStack
from unittest import mock

import pytest

from flliper.srt.environ import envs
from flliper.srt.pdflip import extend_trim as ET
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-pdflip-unit")

MIB = 1 << 20
START = 0.3091        # D_EXTEND_GROWTH_PER_ROW_MIB of the boot (rate_src=start)
CONFIGURED = 16384
FLOOR_CHUNK = 1024    # floor_rows(16384, 1) = 16384 / CAP_FLOOR_DIV
HARD = 638            # near-OOM 400 + F 238


class _Cuda:
    def __init__(self, free):
        self.free = float(free)

    def is_current_stream_capturing(self):
        return False

    def mem_get_info(self, *a):
        return int(self.free * MIB), int(20055 * MIB)

    def memory_reserved(self, *a):
        return int(17000 * MIB)

    def memory_allocated(self, *a):
        return int(16300 * MIB)

    def synchronize(self):
        pass

    def empty_cache(self):
        pass


@pytest.fixture(autouse=True)
def _boot_state():
    """TP2 of the boot: start rate 0.3091 on every rank, Q-710 floor armed, line 1024 UNMEASURED-FALLBACK, no reserve
    (masked to 0 on the D group: stated 1400, effective 0)."""
    with ExitStack() as st:
        st.enter_context(envs.FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB.override(str(START)))
        st.enter_context(envs.FLLIPER_PDFLIP_EXTEND_CAP_FLOOR.override(True))
        st.enter_context(envs.FLLIPER_PDFLIP_EXTEND_RATE_MEASURE.override(True))
        st.enter_context(envs.FLLIPER_PDFLIP_EXTEND_SOFT_LINE.override(True))
        st.enter_context(mock.patch.object(ET, "_read_corridor_line", lambda: (1024.0, "UNMEASURED-FALLBACK:1024")))
        st.enter_context(mock.patch.object(ET, "_user_reserve_mib", lambda: 0.0))
        ET.reset_for_tests()
        yield
        ET.reset_for_tests()


def _vote(free, *, configured=CONFIGURED):
    return ET.width_vote(_Cuda(free), 2, configured, 1, True)


def _measured(rate, rows):
    """This rank's soft measurement as ``measure_close`` leaves it."""
    ET._CACHE["soft_measured"] = rate
    ET._CACHE["soft_rows"] = rows


def test_free_after_the_trim_funds_over_the_plan_line():
    # post 1184 (the boot's first TP2 vote): the 1024 line left cap 516 and the floor chunk 1024 ran; the hard floor
    # funds floor((1184 - 638) / 0.3091) = 1766 rows even at the start rate (no measurement yet)
    assert _vote(1184) == 1766
    assert ET._CACHE["soft_measured"] is None


def test_measured_cost_per_row_widens_what_the_old_rule_never_could():
    # 139 MiB over 1024 rows measured -> 0.1358/row; x 1.25 = 0.1698 (never under 0.5 x start = 0.1546)
    _measured(0.1358, 1024)
    assert ET.soft_base_rate(START, 0.1358) == (0.1698, "measured-soft")
    # free 892 (TP2 at rest): old rule floor chunk 1024; soft floor((892 - 638) / 0.1698) = 1495
    assert _vote(892) == 1495


@pytest.mark.parametrize("measured,expect", [
    (None, (START, "start")),          # nothing measured: the start rate
    (0.05, (0.1546, "measured-soft")),  # one quiet boot: never under half the start rate
    (0.30, (START, "start")),           # 0.30 x 1.25 > start: never over the start rate
])
def test_soft_rate_is_bounded_by_half_the_start_and_the_start(measured, expect):
    assert ET.soft_base_rate(START, measured) == expect


def test_near_oom_free_keeps_the_crawl_floor_not_the_soft_cap():
    # free 420 is under the hard floor 638: the line-respecting cap is one page; the Q-710 floor (crawl guard) holds
    # the width at the floor chunk, never under the cap
    w = _vote(420)
    assert w == FLOOR_CHUNK
    assert ET.rows_cap(420, START, 1, HARD) == 1


def test_user_reserve_is_a_floor_of_the_free_column():
    # reserve 1400 NOT masked (a card without a booked rest): the cap may not take free under 1400 -- the line is the
    # reserve, not 638, and not the reserve added on top of 638. The Q-710 floor chunk (crawl guard, pre-existing)
    # is the one thing that still ignores a line; it is off here to read the cap itself.
    with envs.FLLIPER_PDFLIP_EXTEND_CAP_FLOOR.override(False), mock.patch.object(ET, "_user_reserve_mib", lambda: 1400.0):
        ET.reset_for_tests()
        w = _vote(1500)
    assert w == 323                                    # floor((1500 - 1400) / 0.3091)
    assert 1500 - w * START >= 1400
    ET.reset_for_tests()
    assert _vote(1500) == 2788                         # masked to 0: floor((1500 - 638) / 0.3091)


def test_ramp_guard_holds_a_measured_rate_to_twice_the_widest_measured_extend():
    # a wide chunk can cost more per row than a narrow one (22.10. metal: 2162 rows, reserved +534 MiB for a 282 MiB
    # transient). Only a 1024-row extend was measured: 0.1698 x (1184 - 638) would fund 3215 rows; twice 1024 = 2048
    # holds, the next measurement (at 2048 rows) releases the rest
    _measured(0.1358, 1024)
    assert ET.rows_cap(1184, 0.1698, 1, HARD) == 3215
    assert _vote(1184) == 2048


def test_never_narrower_than_the_old_rule():
    # derived property: soft line <= old line and soft rate <= old rate, so the soft width is never under the old one
    # (reserve 0). The old rule is the same vote with the switch off.
    for measured, rows in ((None, 0), (0.1358, 1024), (0.05, 512), (0.30, 4096)):
        for free in range(380, 3000, 97):
            ET.reset_for_tests()
            if measured is not None:
                _measured(measured, rows)
            soft = _vote(free)
            ET.reset_for_tests()
            with envs.FLLIPER_PDFLIP_EXTEND_SOFT_LINE.override(False):
                old = _vote(free)
            assert soft is None and old is None or (soft is not None and old is not None and soft >= old), \
                (measured, free, soft, old)


def test_switch_off_is_the_old_vote_byte_for_byte(caplog):
    # post 1184 under the 1024 line: cap 516, raised to the floor chunk 1024; no EXTEND-CAP-SOFT line, nothing measured
    ET.reset_for_tests()
    with envs.FLLIPER_PDFLIP_EXTEND_SOFT_LINE.override(False), caplog.at_level(logging.INFO, logger=ET.logger.name):
        assert _vote(1184) == FLOOR_CHUNK
        assert ET.soft_measure_on(2) is False
    assert not [r for r in caplog.records if ET.SOFT_MARKER in r.getMessage()]


def test_one_soft_line_per_width_with_the_fields_of_the_boot_proof(caplog):
    _measured(0.1358, 1024)
    with caplog.at_level(logging.INFO, logger=ET.logger.name):
        _vote(892)
        _vote(892)
        _vote(1184)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith(ET.SOFT_MARKER)]
    assert len(lines) == 2
    first = lines[0]
    for field in ("rank=2", "free=892", "hard_floor=638", "rate_used=0.1698", "rate_src=measured-soft", "cap=1495",
                  "width=1495", "configured=16384"):
        assert field in first, (field, first)


class _Mode:
    def is_extend(self):
        return True

    def is_target_verify(self):
        return False


class _Ids:
    def __init__(self, n):
        self.shape = (n,)


class _AllocCuda:
    """The allocator of PDFLIP-VRAM-PEAK's TP2 window: allocated 16230, the extend raises the peak by its transient."""

    def __init__(self):
        self.peak, self.allocated, self.reserved = 16230, 16230, 17102

    def is_current_stream_capturing(self):
        return False

    def memory_stats(self):
        return {"allocated_bytes.all.peak": self.peak * MIB, "allocated_bytes.all.current": self.allocated * MIB,
                "reserved_bytes.all.current": self.reserved * MIB}


def _extend(cuda, runner, rows, transient_mib, growth_mib=0):
    worker = types.SimpleNamespace(is_draft_worker=False, model_runner=runner)
    ET.measure_open(worker, types.SimpleNamespace(forward_mode=_Mode()), cuda)
    cuda.peak = max(cuda.peak, cuda.allocated + transient_mib)
    cuda.reserved += growth_mib
    return ET.measure_close(runner, types.SimpleNamespace(forward_mode=_Mode(), input_ids=_Ids(rows)), cuda, rank=2)


def test_a_1024_row_chunk_is_measured_where_the_old_measurement_never_looked():
    # the defect: measure_close only counted extends >= 2048 rows, a vote at 1024 rows never measured itself and
    # stayed on rate_src=start for ever
    cuda, runner = _AllocCuda(), types.SimpleNamespace(is_draft_worker=False)
    line = _extend(cuda, runner, 1024, 139, growth_mib=58)
    assert line is not None and "source=measured-soft rank=2 rows=1024" in line
    assert ET._CACHE["soft_measured"] == 0.1358             # ceil4(139 / 1024)
    assert ET._CACHE["soft_rows"] == 1024
    assert ET.measured_rate() is None                       # the >= 2048-row ratchet is untouched
    cuda.peak = cuda.allocated
    assert _extend(cuda, runner, 400, 200) is None     # under SOFT_MIN_ROWS: no price (fixed part)
    assert ET._CACHE["soft_measured"] == 0.1358
    assert _vote(892) == 1495                                # the measured chunk widens the next vote
