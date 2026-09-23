"""fnFL2x37 (23.09.2026): the xsn290 rule ("the card physically holds the
tag above its floor and the peer's staging -> claim OVERDRAWN") was applied
ONCE, at the door of `wait_for`. D TP2 entered with 1669 MiB free for an
868 MiB tag (short by 371 with floor 767 and 471 MiB of peer staging), card 2
then freed 3.2 GB while TP2 sat in the loop reading only the peer's counter
-- 120 s, W35, the whole P<->D lockstep wedged behind that one wait with
room on every card.  The loop now re-reads the card on every pass.
"""
from __future__ import annotations

import os
import threading
import time

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.weg2_memory_saver import (  # noqa: E402
    VramCredit,
    Weg2VramCreditRefused,
)

MIB = 1 << 20


def _short_counter(tmp_path, name):
    """x37's book on card 2 at the door: the peer funded 600 and staged 471
    of it (balance 129 for an 868 MiB tag); free 1669 - floor 767 - staged
    471 = 431 < 868, so the entry exit does NOT fire and the loop is entered."""
    c = VramCredit(name, credit_dir=str(tmp_path))
    c.begin_leg("flip-2")
    c.publish("weights_13", 600 * MIB)
    assert c.debit("ipc-stage", 471 * MIB)
    return c


def test_a_card_that_frees_up_inside_the_wait_is_granted_overdrawn(tmp_path):
    c = _short_counter(tmp_path, "GPU-x37-a")
    state = {"free": 1669 * MIB}          # x37's entry reading: short

    def _p_pauses_its_kv():
        time.sleep(0.3)
        state["free"] = 4864 * MIB        # nvidia-smi at 08:47: 4.9 GB free

    threading.Thread(target=_p_pauses_its_kv).start()
    t0 = time.perf_counter()
    rec = c.wait_for(868 * MIB, budget_s=5.0, tag="weights_6", free_bytes_now=state["free"],
                     free_reader=lambda: state["free"], floor_bytes=767 * MIB,
                     epoch="flip-2", poll_s=0.05)
    assert time.perf_counter() - t0 < 3.0
    assert rec["claimed_bytes"] == 868 * MIB
    assert "OVERDRAWN" in rec["reason"] and "re-read in the loop" in rec["reason"]
    assert rec["free_bytes"] == 4864 * MIB


def test_a_card_that_stays_short_still_waits_out_the_budget(tmp_path):
    """The mutant's other half: room that never comes is still a W35, never
    a grant on the counter alone."""
    c = _short_counter(tmp_path, "GPU-x37-b")
    with pytest.raises(Weg2VramCreditRefused):
        c.wait_for(868 * MIB, budget_s=0.6, tag="weights_6", free_bytes_now=1669 * MIB,
                   free_reader=lambda: 1669 * MIB, floor_bytes=767 * MIB,
                   epoch="flip-2", poll_s=0.05)
    assert c.read()["consumed_bytes"] == 471 * MIB   # only the peer's staging, nothing claimed
