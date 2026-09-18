"""weg2xsn274 (18.09.2026): group D died again in flip 2 -- TP0 (5090) took
`wait_for`'s early exit ("the card already holds the bytes") with
free=3156 MiB for a 2502 MiB tag while the measured corridor floor was
767 MiB (WEG2-CREDIT-FLOOR MEASURED-D). The exit graded free >= need
without the floor; cu_mem_create ran out of memory, exit(1). The gate
inside the loop grades `free - floor` (allocatable); the early exit must
grade the same term.
"""

from __future__ import annotations

import os
import threading
import time

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.weg2_memory_saver import (  # noqa: E402
    VramCredit,
    Weg2VramCreditAllocatableShort,
)

MIB = 1 << 20


def _credit(tmp_path, name):
    c = VramCredit(name, credit_dir=str(tmp_path))
    c.begin_leg("flip-2")
    return c


def test_the_early_exit_grades_free_minus_floor_not_free_alone(tmp_path):
    c = _credit(tmp_path, "GPU-xsn274-a")
    c.publish("weights_0", 7700 * MIB)        # the peer's releases cover the tag
    need = 2502 * MIB
    state = {"free": 3156 * MIB}

    # xsn274's numbers: free >= need but free - floor < need. Not the early
    # exit any more; the gate polls the allocatable estimate on a short bound
    # and refuses W85 by name when the card stays short -- never an OOM.
    with pytest.raises(Weg2VramCreditAllocatableShort):
        c.wait_for(need, budget_s=5.0, tag="weights_3", free_bytes_now=state["free"],
                   free_reader=lambda: state["free"], floor_bytes=767 * MIB,
                   epoch="flip-2", alloc_poll_s=0.3)
    assert c.read()["consumed_bytes"] == 0          # nothing was debited


def test_with_room_above_the_floor_the_early_exit_still_holds(tmp_path):
    c = _credit(tmp_path, "GPU-xsn274-b")
    c.publish("weights_0", 7700 * MIB)
    rec = c.wait_for(2502 * MIB, budget_s=5.0, tag="weights_3", free_bytes_now=3500 * MIB,
                     free_reader=lambda: 3500 * MIB, floor_bytes=767 * MIB, epoch="flip-2")
    assert rec["waited_s"] == 0.0 and rec["claimed_bytes"] == 2502 * MIB
    assert rec["corridor_floor_bytes"] == 767 * MIB
    assert rec["allocatable_est_bytes"] == (3500 - 767) * MIB
    assert "above its corridor floor" in rec["reason"]


def test_the_card_that_frees_up_during_the_poll_is_granted(tmp_path):
    c = _credit(tmp_path, "GPU-xsn274-c")
    c.publish("weights_0", 7700 * MIB)
    state = {"free": 3156 * MIB}

    def _sleeper_pauses_more():
        time.sleep(0.2)
        state["free"] = 6000 * MIB               # PP0 paused another tag

    threading.Thread(target=_sleeper_pauses_more).start()
    rec = c.wait_for(2502 * MIB, budget_s=5.0, tag="weights_3", free_bytes_now=state["free"],
                     free_reader=lambda: state["free"], floor_bytes=767 * MIB,
                     epoch="flip-2", alloc_poll_s=3.0)
    assert rec["claimed_bytes"] == 2502 * MIB and rec["waited_s"] >= 0.15


def test_xsn284_a_live_but_short_counter_sends_the_tag_into_the_loop(tmp_path):
    """xsn284: free=4338, floor=767, need=2502 -- allocatable covered -- but the
    peer's counter had 741 MiB left; the peer refunded and re-staged 1673 MiB
    into that free between the reading and cu_mem_create. The early exit now
    needs the counter to COVER the tag; short, the tag waits on the counter
    and is granted when the peer publishes."""
    c = _credit(tmp_path, "GPU-xsn284-a")
    c.publish("weights_3", 741 * MIB)               # a live counter, short
    state = {"free": 4338 * MIB}

    def _peer_pauses_next_tag():
        time.sleep(0.25)
        c.publish("weights_4", 2502 * MIB)

    threading.Thread(target=_peer_pauses_next_tag).start()
    rec = c.wait_for(2502 * MIB, budget_s=5.0, tag="weights_4", free_bytes_now=state["free"],
                     free_reader=lambda: state["free"], floor_bytes=767 * MIB,
                     epoch="flip-2", alloc_poll_s=3.0)
    assert rec["waited_s"] >= 0.2 and rec["claimed_bytes"] == 2502 * MIB
    assert c.read()["consumed_bytes"] == 2502 * MIB


def test_xsn284_a_single_group_boot_without_a_counter_still_takes_the_early_exit(tmp_path):
    c = VramCredit("GPU-xsn284-b", credit_dir=str(tmp_path))   # no leg opened: no counter
    rec = c.wait_for(2502 * MIB, budget_s=1.0, tag="weights_4", free_bytes_now=4338 * MIB,
                     free_reader=lambda: 4338 * MIB, floor_bytes=767 * MIB, epoch="flip-2")
    assert rec["waited_s"] == 0.0 and "no peer counter" in rec["reason"]


def test_xsn284_a_covering_counter_takes_the_early_exit_and_names_it(tmp_path):
    c = _credit(tmp_path, "GPU-xsn284-c")
    c.publish("weights_3", 7700 * MIB)
    rec = c.wait_for(2502 * MIB, budget_s=1.0, tag="weights_4", free_bytes_now=4338 * MIB,
                     free_reader=lambda: 4338 * MIB, floor_bytes=767 * MIB, epoch="flip-2")
    assert rec["waited_s"] == 0.0 and rec["claimed_bytes"] == 2502 * MIB
    assert "counter covers the tag" in rec["reason"]
