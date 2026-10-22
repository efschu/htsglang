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
