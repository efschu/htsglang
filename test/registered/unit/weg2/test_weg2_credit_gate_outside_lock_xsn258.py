"""weg2xsn258 (17.09.2026): the credit wait's allocatable poll ran INSIDE
`VramCredit.claim`'s exclusive flock.

Measured on the metal (boot weg2xsn258, first flip D->P):

    D-TP0  WEG2-SLEEP-TAG-TIME tag=weights_1 deposit_ms=21 pause_ms=16
           credit_ms=30008
    PP0    WEG2-RESUME begin tag=weights_1 need_mib=3820 free_mib=1718
           ... no credit-ok ever; W35 after the 120 s budget

PP0 waited for weights_1 with the card short (free < need), so its grant
gate polled the NVML free column for up to 30 s -- holding the counter
file's LOCK_EX the whole time. D-TP0's `publish` of that very tag (the
release that would have funded PP0) blocked behind that lock for 30 008 ms.
When it finally landed, PP0's poll loop had already spent its 30 s
allocatable bound on a reading that could not change while the publisher
was locked out, and the wait never recovered.

THE FIX: the gate raises `_AllocatableTransient` while the bounded poll has
time left; `claim` writes nothing on it; `wait_for` sleeps OUTSIDE the lock
and re-tries. The lock is held for the ms of one read.

DANGER DIRECTION (pinned by the mutant below): a gate that sleeps inside
the lock starves the publisher. The publish latency is the measurement.
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


def test_a_publish_is_not_blocked_by_the_waiters_allocatable_poll(tmp_path):
    """The xsn258 shape: W waits with the card short; S publishes the tag
    that funds W ~0.4 s into the wait; the card's free column follows the
    publish. The publish must land in ms, not after the poll's bound, and W
    must then be granted."""
    credit = VramCredit("GPU-test-xsn258", credit_dir=str(tmp_path))
    credit.begin_leg("flip-1")
    need = 3820 * MIB
    state = {"free": 1718 * MIB}
    publish_ms = {"v": None}

    def _free_reader():
        return state["free"]

    def _publisher():
        time.sleep(0.4)
        t0 = time.perf_counter()
        credit.publish("weights_1", 4000 * MIB)
        publish_ms["v"] = (time.perf_counter() - t0) * 1000
        state["free"] = 6000 * MIB

    th = threading.Thread(target=_publisher, daemon=True)
    th.start()
    rec = credit.wait_for(
        need, budget_s=20.0, tag="weights_1", free_bytes_now=state["free"],
        free_reader=_free_reader, floor_bytes=0, epoch="flip-1",
        alloc_poll_s=10.0,
    )
    th.join(timeout=5.0)
    assert publish_ms["v"] is not None, "the publisher never ran"
    assert publish_ms["v"] < 1000, (
        f"publish blocked {publish_ms['v']:.0f} ms behind the waiter's lock "
        f"-- the xsn258 starvation")
    assert int(rec["claimed_bytes"]) >= need
    assert rec["waited_s"] < 5.0


def test_a_genuine_shortfall_still_refuses_w85_after_the_bound(tmp_path):
    """The bound moved out of the lock but did not disappear: with credit
    published and the card genuinely short, W85 is still raised, by name,
    once `alloc_poll_s` has elapsed."""
    credit = VramCredit("GPU-test-xsn258b", credit_dir=str(tmp_path))
    credit.begin_leg("flip-1")
    credit.publish("weights_1", 4000 * MIB)
    need = 3820 * MIB
    t0 = time.perf_counter()
    with pytest.raises(Weg2VramCreditAllocatableShort) as exc:
        credit.wait_for(
            need, budget_s=20.0, tag="weights_1", free_bytes_now=100 * MIB,
            free_reader=lambda: 100 * MIB, floor_bytes=0, epoch="flip-1",
            alloc_poll_s=0.3,
        )
    elapsed = time.perf_counter() - t0
    assert 0.25 <= elapsed < 5.0, f"bound not honoured: {elapsed:.2f}s"
    assert "W85" in str(exc.value)
    # nothing was debited by the refused claims
    assert int(credit.read().get("consumed_bytes", 0)) == 0


def test_a_transient_shortfall_that_clears_is_granted_and_debited_once(tmp_path):
    """xsn108's own case (the sleeper's IPC staging freed a moment later):
    the card is short for ~0.3 s, then covers the request -- granted, and
    the debit lands exactly once."""
    credit = VramCredit("GPU-test-xsn258c", credit_dir=str(tmp_path))
    credit.begin_leg("flip-1")
    credit.publish("weights_1", 4000 * MIB)
    need = 2000 * MIB
    t_start = time.perf_counter()

    def _free_reader():
        return (100 * MIB) if time.perf_counter() - t_start < 0.3 else (5000 * MIB)

    rec = credit.wait_for(
        need, budget_s=20.0, tag="weights_1", free_bytes_now=100 * MIB,
        free_reader=_free_reader, floor_bytes=0, epoch="flip-1",
        alloc_poll_s=10.0,
    )
    assert int(rec["claimed_bytes"]) == need
    assert int(credit.read()["consumed_bytes"]) == need
