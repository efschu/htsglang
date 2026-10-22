"""Review of EXTEND-KORRIDOR (desk/nf-rc33-extendkorridor-1022 @ bc9dcc7520), adversarial, 22.10.

What this module holds beyond the author's test_nf_rc33_extendkorridor_1022.py:

(R1) SLEEP POISON (danger direction: the width vote's rate ratchets on a number that is not an extend's cost).
     Since EXTEND-KORRIDOR ``lend_rate`` -- the LEND-COST per-row cost of a wide extend -- is a term of the #794
     width vote, ratchet up only, for the rest of the boot. Its draft term is ``reserved_at_next_pass -
     reserved_after_target``. An extend right before a sleep (R6 flips mid-D-prefill) left the tail open; the
     "next pass" was then the first pass after the WAKE, and every byte the wake put into torch's pool on top of
     the pre-sleep reserved became draft growth: +3000 MiB on a 2710-row extend -> 1.85 MiB/row -> every later
     chunk capped to a few hundred rows. ``d_transient_lend.quiesce_for_sleep`` now closes the tail before the
     pause, with the extend batch's own reserved.
(R2) ``envs.X.override`` restores in ``finally``: a red test body inside the ``with`` no longer leaves the
     variable set for every following module (the author's first HEAD run: 22 follow-up failures).
(R3) R11: the vote stays a group MIN of rank-local caps -- the group width funds every rank's card down to its
     own line at its own rate (each rank's rate and line are its own; no rank adopts a peer's reading).
"""
from __future__ import annotations

import math
import os
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import d_transient_lend as dtl  # noqa: E402
from flliper.srt.pdflip import extend_trim as ET  # noqa: E402

MIB = 1 << 20
PAGE = 64
START = 0.5871
LINES = {0: 767.0, 1: 700.0, 2: 701.0}


@pytest.fixture(autouse=True)
def _fresh():
    ET.reset_for_tests()
    with envs.FLLIPER_PDFLIP_EXTEND_RATE_MEASURE.override(True), \
            envs.FLLIPER_PDFLIP_EXTEND_RATE_MEASURE_RANKS.override("1,2"), \
            envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override("767,700,701"), \
            envs.FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB.override("0.5871,0.5871,0.5871"), \
            mock.patch.object(ET, "_read_corridor_line", lambda: (767.0, "MEASURED-D:767")):
        yield
    ET.reset_for_tests()


class Cuda:
    """torch.cuda stand-in: driver free and torch reserved in MiB."""

    def __init__(self, free, reserved):
        self.free, self.reserved = float(free), float(reserved)
        self.trims = 0

    def is_current_stream_capturing(self):
        return False

    def mem_get_info(self):
        return int(self.free * MIB), int(32088 * MIB)

    def memory_reserved(self):
        return int(self.reserved * MIB)

    def memory_stats(self):
        return {"allocated_bytes.all.peak": 0, "allocated_bytes.all.current": 0,
                "reserved_bytes.all.current": int(self.reserved * MIB)}

    def synchronize(self):
        pass

    def empty_cache(self):
        self.trims += 1


def _open_tail(rows=2710, growth_mib=2024, r1_mib=33116.0):
    """The state measure_close leaves after a target extend: the tail waits for the next pass's settle."""
    ET._CACHE["lend_cost_tail"] = (rows, int(1416 * MIB), int(growth_mib * MIB), int(r1_mib * MIB))


def _sched(rank=0):
    return types.SimpleNamespace(ps=types.SimpleNamespace(tp_rank=rank))


def _patch_torch_cuda(cuda):
    import torch

    return mock.patch.multiple(torch.cuda, memory_stats=cuda.memory_stats, mem_get_info=cuda.mem_get_info,
                               memory_reserved=cuda.memory_reserved, synchronize=cuda.synchronize,
                               empty_cache=cuda.empty_cache,
                               is_current_stream_capturing=cuda.is_current_stream_capturing)


def _quiet_bank():
    from flliper.srt.pdflip import d_row_fence
    from flliper.srt.pdflip import d_seat_vram as dsv

    return mock.patch.object(dsv, "controller", lambda sched: None), \
        mock.patch.object(d_row_fence, "quiesce_warm", lambda: None)


# ---- (R1) the sleep closes the tail with the extend's own reserved ------------------------------------

def test_sleep_closes_the_lend_cost_tail_before_the_pause():
    _open_tail()
    pre_sleep = Cuda(free=900.0, reserved=33116.0)             # the draft extend grew nothing
    p1, p2 = _quiet_bank()
    with _patch_torch_cuda(pre_sleep), p1, p2:
        dtl.quiesce_for_sleep(_sched(0))
    assert ET._CACHE["lend_cost_tail"] is None
    assert ET.lend_rate(0) == pytest.approx(math.ceil(2024 / 2710 * 1e4) / 1e4)   # 0.7469: the extend's cost


def test_after_the_wake_nothing_is_left_to_book_as_draft_growth():
    _open_tail()
    p1, p2 = _quiet_bank()
    with _patch_torch_cuda(Cuda(free=900.0, reserved=33116.0)), p1, p2:
        dtl.quiesce_for_sleep(_sched(0))
    honest = ET.lend_rate(0)
    # the wake put 3000 MiB more into torch's pool than the extend batch left (a larger KV stage, say)
    woken = Cuda(free=2071.0, reserved=36116.0)
    assert ET.lend_cost_settle(0, woken) is None              # no tail: nothing booked
    assert ET.lend_rate(0) == pytest.approx(honest)
    cap = ET.width_vote(woken, 0, 4096, PAGE, True)
    assert cap == (int((2071.0 - 767.0) / honest) // PAGE) * PAGE   # 1728: the honest rate, not 1.85 MiB/row


def test_what_the_open_tail_would_have_booked_after_the_wake():
    # the danger direction, stated as numbers: a tail settled after the wake reads the wake's pool
    _open_tail()
    woken = Cuda(free=2071.0, reserved=36116.0)
    ET.lend_cost_settle(0, woken)
    poisoned = ET.lend_rate(0)
    assert poisoned == pytest.approx(math.ceil((2024 + 3000) / 2710 * 1e4) / 1e4)  # 1.8539 MiB/row
    assert ET.width_vote(woken, 0, 4096, PAGE, True) == 640    # every later chunk: 640 rows instead of 1728


def test_sleep_without_an_open_tail_moves_nothing():
    p1, p2 = _quiet_bank()
    cuda = Cuda(free=500.0, reserved=33116.0)
    with _patch_torch_cuda(cuda), p1, p2:
        dtl.quiesce_for_sleep(_sched(0))
    assert cuda.trims == 0 and ET.lend_rate(0) is None


# ---- (R2) override restores after an exception ---------------------------------------------------------

def test_override_restores_when_the_body_raises():
    var = envs.FLLIPER_PDFLIP_EXTEND_TRIM_MIB
    before_present = var.name in os.environ
    before = os.environ.get(var.name)
    with pytest.raises(RuntimeError):
        with var.override("1,2,3"):
            assert os.environ[var.name] == "1,2,3"
            raise RuntimeError("a red test body")
    assert (var.name in os.environ) == before_present
    assert os.environ.get(var.name) == before


# ---- (R3) R11: the group width funds every rank at its own line and rate ------------------------------

@pytest.mark.parametrize("posts", [(2071, 1820, 1700), (2071, 2400, 2300), (1500, 2500, 900)])
def test_group_min_funds_every_rank_at_its_own_line_and_rate(posts):
    # each rank is its own process: its own lend_rate / Q-694b reading and its own corridor line
    readings = {0: {"lend": 0.7547, "meas": None}, 1: {"lend": 0.5164, "meas": 0.5164},
                2: {"lend": 0.5431, "meas": 0.5514}}
    votes, rates = {}, {}
    for rank in (0, 1, 2):
        ET.reset_for_tests()
        ET._CACHE["lend_rate"] = readings[rank]["lend"]
        ET._CACHE["measured"] = readings[rank]["meas"]
        line = LINES[rank]
        with mock.patch.object(ET, "_read_corridor_line", lambda line=line: (line, "MEASURED-D")):
            votes[rank] = ET.width_vote(Cuda(free=float(posts[rank]), reserved=31092.0), rank, 4096, PAGE, True)
            rates[rank] = ET.effective_rate(START, rank)
    group = min(4096 if v is None else v for v in votes.values())
    for rank in (0, 1, 2):
        if group > PAGE:
            assert posts[rank] - group * rates[rank] >= LINES[rank] - 1e-6, (rank, group, rates[rank])


# ---- (R4) a card at or under the line votes ONE PAGE, never 0 ------------------------------------------
# The consumer (scheduler ``_corridor_granted_prefill_width``) reads a reduced width <= 0 as "do not narrow":
# a vote of 0 would let the full configured chunk run on a card already at the law. With the line at 767
# instead of 300 the card sits under ``line + rate x page`` far more often (the lend fills down to the floor).

@pytest.mark.parametrize("post", [500.0, 767.0, 800.0, 804.0])
def test_card_at_the_line_votes_one_page_not_zero(post):
    ET._CACHE["lend_rate"] = 0.7547
    assert ET.width_vote(Cuda(free=post, reserved=31092.0), 0, 4096, PAGE, True) == PAGE


# ---- (R5) planner 22.10.: the short-extend price is rc33 byte for byte; the vote cuts at the corridor -------
# A4 of 21.10. stands: the price keeps the 300 MiB HALT line and the rc33 rate (start / Q-694b x 1.15). Neither the
# corridor line nor the wide-extend cost (lend_rate, its recorded start) may move it.

def _rc33_price(rank, rows, floor, rate, measured_rate_rank, cost=None):
    if cost is not None:
        return max(floor, 300.0 + rate * rows, 300.0 + cost)
    if not measured_rate_rank:
        return floor + 300.0 + rate * rows
    return max(floor, 300.0 + rate * rows)


@pytest.mark.parametrize("rows", [1, 5, 64, 129, 512, 900, 1400])
@pytest.mark.parametrize("lend", [None, 0.7547, 1.2])
def test_short_price_is_rc33_whatever_the_vote_knows(rows, lend):
    with envs.FLLIPER_PDFLIP_EXTEND_LEND_RATE_START_MIB.override("0.7547,0,0"):
        ET.reset_for_tests()
        if lend is not None:
            ET._CACHE["lend_rate"] = lend
        assert ET.vote_rate(START, 0) >= 0.7547                       # the vote knows the wide cost
        assert ET.corridor_line_mib(0) == 767.0                        # and the corridor line
        assert dtl.short_extend_need_mib(0, rows) == pytest.approx(_rc33_price(0, rows, 767.0, START, False))
        assert ET.chunk_need_mib(0, rows) == pytest.approx(300.0 + START * rows)


def test_short_price_with_a_class_reading_is_rc33():
    with envs.FLLIPER_PDFLIP_EXTEND_LEND_RATE_START_MIB.override("0.7547,0,0"):
        ET.reset_for_tests()
        ET._CACHE["lend_cost"] = {1024: (742, 901)}
        ET._CACHE["lend_rate"] = 0.7547
        assert dtl.short_extend_need_mib(0, 900) == pytest.approx(_rc33_price(0, 900, 767.0, START, False, 742))
        assert dtl.short_extend_need_mib(0, 900) == pytest.approx(1042.0)   # 06:17:02: kept on 1094, A4


def test_short_price_measured_rate_rank_is_rc33():
    ET._CACHE["measured"] = 0.5164                                     # TP1 Q-694b
    ET._CACHE["lend_rate"] = 0.9                                       # a wide reading may not leak into the price
    rate = max(START, math.ceil(0.5164 * 1.15 * 1e4) / 1e4)
    assert dtl.short_extend_need_mib(1, 600) == pytest.approx(_rc33_price(1, 600, 700.0, rate, True))
    assert ET.vote_rate(START, 1) == pytest.approx(0.9)


# ---- (R6) planner 22.10.: TP0 starts the vote at the recorded wide-extend cost --------------------------------

def test_the_recorded_start_prices_the_first_wide_extend_after_the_boot():
    # rc33 06:14:15: TP0 post 2137, no reading yet; the 2880-row extend cost 0.6702 MiB/row (1930 MiB)
    with envs.FLLIPER_PDFLIP_EXTEND_LEND_RATE_START_MIB.override("0.7547,0,0"):
        ET.reset_for_tests()
        assert ET.lend_rate(0) == pytest.approx(0.7547)
        cap = ET.width_vote(Cuda(free=2137.0, reserved=31092.0), 0, 4096, PAGE, True)
        assert cap == 1792                                             # floor((2137 - 767) / 0.7547) = 1815 -> page
        assert 2137.0 - cap * (1930 / 2880) >= 767.0                   # ends ~936, not ~593 (start 0.5871: cap 2304)
        assert 2137.0 - cap * 0.7547 >= 767.0


def test_the_start_is_a_floor_the_boots_readings_ratchet_over():
    with envs.FLLIPER_PDFLIP_EXTEND_LEND_RATE_START_MIB.override("0.7547,0,0"):
        ET.reset_for_tests()
        ET._CACHE["lend_rate"] = 0.70                                  # a cheaper own reading never lowers it
        assert ET.lend_rate(0) == pytest.approx(0.7547)
        ET._CACHE["lend_rate"] = 0.80
        assert ET.lend_rate(0) == pytest.approx(0.80)


def test_the_3080_ranks_get_no_recorded_start():
    with envs.FLLIPER_PDFLIP_EXTEND_LEND_RATE_START_MIB.override("0.7547,0,0"):
        ET.reset_for_tests()
        assert ET.lend_rate(1) is None and ET.lend_rate(2) is None
        assert ET.vote_rate(START, 1) == pytest.approx(START)


def test_no_start_where_lend_cost_is_not_armed():
    with envs.FLLIPER_PDFLIP_EXTEND_LEND_RATE_START_MIB.override("0.7547,0,0"):
        ET.reset_for_tests()
        with mock.patch.object(ET, "lend_cost_armed", lambda rank=None: False):
            assert ET.lend_rate(0) is None


def test_the_launcher_writes_the_start_from_the_nextflash_record():
    from flliper.srt.pdflip import launcher as L

    rec, src = L.d_extend_lend_rate_record("nextflash")
    assert rec == [0.7547, None, None] and "dkrnfint4h6ablxcw3spmlrhlvwflbar1dauer10220606" in src
    ns = types.SimpleNamespace(profile="nextflash", env_d="")
    lines = []
    assert L.d_extend_lend_rate_start_env(ns, lines.append, "D", 3) == "0.7547,0,0"
    assert L.parse_group_env(ns.env_d)[L.EXTEND_LEND_RATE_START_ENV] == "0.7547,0,0"
    assert "Vorrang" not in lines[-1]
    # a second solve pass sees its own write: not a user value
    L.d_extend_lend_rate_start_env(ns, lines.append, "D", 3)
    assert "Vorrang" not in lines[-1]
    # a user value wins
    ns2 = types.SimpleNamespace(profile="nextflash", env_d=f"{L.EXTEND_LEND_RATE_START_ENV}=0.8,0,0")
    assert L.d_extend_lend_rate_start_env(ns2, lines.append, "D", 3) is None
    assert L.parse_group_env(ns2.env_d)[L.EXTEND_LEND_RATE_START_ENV] == "0.8,0,0" and "Vorrang" in lines[-1]
    # another inventory length writes nothing
    ns3 = types.SimpleNamespace(profile="nextflash", env_d="")
    assert L.d_extend_lend_rate_start_env(ns3, lines.append, "D", 2) is None and ns3.env_d == ""
    # the 27B profile has no record: nothing
    ns4 = types.SimpleNamespace(profile="qwen27b", env_d="")
    assert L.d_extend_lend_rate_start_env(ns4, lines.append, "D", 3) is None and ns4.env_d == ""
