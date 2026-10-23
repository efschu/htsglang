# SPDX-License-Identifier: Apache-2.0
"""ROOMSHORT two-stage repro (27B INT4 AutoRound flip, rc35i4ar = a45112428e, boot ..._a45112428e_1023_140032,
P = PP2 on 2x 3080, tp 1 per stage). SOLL v2 R11: the stages receive a finished verdict, no stage decides alone.

THE DEATH (P.log 66782-66829). pdflip-23-212 (prompt 118727, host hit 78271 to load back):

  PP0 14:33:39  SF LOADBACK-ROOM SAME-PASS kv=78271 floor=38395 avail=38395 evictable=89453 evicted=39876 -> loads
                PDFLIP INTAKE-RESUMED ... cap_src=fact(1,91,91) cap=146025   (the PR room vote: PP1's fact)
  PP1 14:33:39  PDFLIP PUBLISH-SWEEP ... issued=1 in_flight_after=1   (the 22-211 tail write-through: write_backup
                inc_lock_ref's the whole 22-210/22-211 path -> ~103.8k rows leave evictable_size)
  PP1 14:33:43  #1465 WRITE-BACK DRAIN n=1 ms=3730   (the row's own evict drains it, the path is evictable again)
  PP1 14:33:43  LB-VERDICT ROW-ROOM kv=78271 avail=38395 evictable=1902 evicted=1902 avail_after=40297 ok=False
                -> AgreedRoomShort -> RANK-DEATH, PP0 FORWARD-HANG, W17

So the verdict WAS stage-agreed: PP0 decided on MIN(own room, PP1's fact room = available + payable = 146025), the
PR vote's R_m. PP1's fact was true -- one peel (with its drain) pays it. What broke is the executing stage: ``row_room``
took ONE read of ``evictable_size()`` before the eviction whose own drain releases the write-through lock, and judged
the row from that read. The rounds (``FLLIPER_PDFLIP_LBV_ROW_ROOM_REREAD``, Q-640's re-read on the row-executing
follower) make the follower pay the room its fact promised.

Two simulated stages share the request; PP0 runs the real ``RoomBook.cap`` + ``set_cap`` + ``local_pp_room`` on a
pool like its own, PP1 runs the real ``row_room`` on a pool whose path is held by an in-flight write-through that its
evict's drain releases. RED on 80268e7d04 (PP1 ok=False), GREEN with the rounds.

MUTANTS (the danger directions): (M1) the single read of before (switch 0) -> the death; (M2) PP0 deciding on its own
room only (the agreed R_m ignored) -> PP0 loads a row that a genuinely short PP1 cannot pay.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest import mock

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import pp_loadback_verdict as LBV  # noqa: E402
from flliper.srt.pdflip import pp_room_vote as PR  # noqa: E402
from flliper.srt.pdflip import pp_slot_fidelity as SF  # noqa: E402

RID = "pdflip-23-212"
KV = 78271
AVAIL = 38395            # both stages (same rows per stage)
PP0_EVICTABLE = 89453
PP1_FIRST_READ = 1902    # PP1's evictable while the 22-211 tail write-through holds the path
PP1_PATH = 105728 - 1902  # rows the write-through lock holds until the drain
PP1_FACT_PAYABLE = 107630  # PP1's fact at its pass top (fact(1,91,91): room 146025 = 38395 + 107630)


class _Alloc:
    def __init__(self, avail):
        self.avail = int(avail)

    def available_size(self):
        return self.avail


class _Stage:
    """One stage's tree: ``held`` rows are behind an in-flight write-through (write_backup's inc_lock_ref on the
    ancestors) and become evictable when an eviction ends in ``writing_check(write_back=True)`` (#1465)."""

    def __init__(self, avail, evictable, held=0):
        self.token_to_kv_pool_allocator = _Alloc(avail)
        self._evictable = int(evictable)
        self._held = int(held)
        self.evicted = []
        setattr(self, SF.FLOOR_LOCAL_PP_ATTR, True)

    def evictable_size(self):
        return self._evictable

    def evict(self, params):
        n = min(int(params.num_tokens), self._evictable)
        self._evictable -= n
        self.token_to_kv_pool_allocator.avail += n
        self.evicted.append(n)
        self._evictable += self._held  # the drain at the end of evict() releases the write-through lock
        self._held = 0
        return SimpleNamespace(num_tokens_evicted=n)


def _pp0_decides(pp0: _Stage, fact_room: int, fact_payable: int) -> bool:
    """PP0's pass: absorb PP1's fact, R_m = MIN over the stages (the real RoomBook), publish it on its tree, then the
    load-back verdict of ``load_back``'s S48 path (``kv <= floor`` else ``local_pp_room``)."""
    book = PR.RoomBook()
    book.begin_pass()
    book.absorb([PR.PdFlipPpRoomFact(rank=1, executed=91, room=int(fact_room), available=AVAIL,
                                     payable=int(fact_payable), reported=PP1_FIRST_READ, processed=91)])
    verdict = book.cap()
    PR.set_cap(pp0, None if verdict is None else verdict.cap)
    floor = SF.agreed_floor(pp0, AVAIL)
    if KV <= floor:
        return True
    return bool(SF.local_pp_room(pp0, KV, floor, RID))


def _pp1_executes(pp1: _Stage):
    """PP1 executes PP0's row (``row_room_applies``): no room verdict of its own -- it makes the room or stops."""
    return LBV.row_room(pp1, KV, RID)


@pytest.fixture(autouse=True)
def _env():
    with mock.patch.dict(os.environ, {}, clear=False):
        for k in ("FLLIPER_PDFLIP_LBV_ROW_ROOM_REREAD", "FLLIPER_PDFLIP_DUAL_LAYOUT", SF.ENV):
            os.environ.pop(k, None)
        yield


def test_boot_1023_140032_both_stages_load_the_row_pp0_decided():
    """THE DEATH, red on 80268e7d04: PP0 decides on the agreed R_m (146025 from PP1's fact) and loads; PP1, whose path
    a write-through holds at the read, must still pay the row -- its own evict's drain makes the room."""
    pp0 = _Stage(AVAIL, PP0_EVICTABLE)
    assert _pp0_decides(pp0, AVAIL + PP1_FACT_PAYABLE, PP1_FACT_PAYABLE) is True
    assert pp0.token_to_kv_pool_allocator.avail >= KV

    pp1 = _Stage(AVAIL, PP1_FIRST_READ, held=PP1_PATH)
    ok, a0, ev0, evd, a1 = _pp1_executes(pp1)
    assert ok is True, f"PP1 cannot pay the row PP0 loaded: avail={a0} evictable={ev0} evicted={evd} after={a1}"
    assert a1 >= KV and evd == KV - AVAIL  # never more than the shortfall


def test_mutant_single_read_is_the_death():
    """M1: the one-read row_room of before (switch 0) reproduces the boot's line byte for byte in its numbers."""
    with mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_LBV_ROW_ROOM_REREAD": "0"}):
        pp1 = _Stage(AVAIL, PP1_FIRST_READ, held=PP1_PATH)
        ok, a0, ev0, evd, a1 = _pp1_executes(pp1)
    assert (ok, a0, ev0, evd, a1) == (False, 38395, 1902, 1902, 40297)


def test_danger_direction_no_stage_loads_against_a_short_peer():
    """A follower that genuinely cannot pay (its fact: room 40297 < 78271) -> PP0 refuses this pass (the request waits,
    ``LB-VERDICT PP0-WAIT``) although its OWN room would hold it. Nobody loads, nobody dies, nobody splits."""
    pp0 = _Stage(AVAIL, PP0_EVICTABLE)
    assert _pp0_decides(pp0, AVAIL + PP1_FIRST_READ, PP1_FIRST_READ) is False
    assert pp0.evicted == [] and pp0.token_to_kv_pool_allocator.avail == AVAIL  # no eviction for a refused load


def test_mutant_pp0_on_its_own_room_alone_is_caught():
    """M2: PP0 ignoring the agreed R_m (deciding on its own room, the rank-local verdict R11 forbids) loads the row
    the short follower then cannot pay -- the pair the test above forbids."""
    pp0 = _Stage(AVAIL, PP0_EVICTABLE)
    with mock.patch.object(SF, "agreed_room", lambda tree: None), \
            mock.patch.object(SF, "agreed_floor", lambda tree, floor: int(floor)):
        loaded = _pp0_decides(pp0, AVAIL + PP1_FIRST_READ, PP1_FIRST_READ)
    pp1 = _Stage(AVAIL, PP1_FIRST_READ, held=0)
    ok = _pp1_executes(pp1)[0]
    assert loaded is True and ok is False  # the mutant's split: PP0 loaded, PP1 short -> the named group stop


def test_true_residual_on_the_follower_stays_a_named_stop_without_spin():
    """What no peel can pay stays the named stop (raenge-nie-uneins): one round that frees what it sees, then the
    loop ends because nothing more is freed or released."""
    pp1 = _Stage(AVAIL, PP1_FIRST_READ, held=0)
    ok, _a0, _ev0, _evd, a1 = _pp1_executes(pp1)
    assert ok is False and a1 == AVAIL + PP1_FIRST_READ and pp1.evicted == [PP1_FIRST_READ]
