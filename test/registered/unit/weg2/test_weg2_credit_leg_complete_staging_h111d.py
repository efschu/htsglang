"""H111d (26.09.2026, boot fnFL2h91bb1, first P->D flip, D TP2 on card 2).

D TP2 asked for ``weights_8`` (1000 MiB) at 14:17:52.75 with 1049 MiB free;
the co-located sleeper P PP2 ended its tag loop at 14:17:53.086, published
``leg_complete`` and only THEN ran its leg-end drain, which frees its two
on-card c2 IPC stagings (2 x 617 MiB, booked on this very counter). TP2's
pass at 14:17:53.089 read free 1699 - floor 701 = 998 < 1000 with
``peer_leg_complete=True`` and refused (W35), 1234 MiB before the staging
came back. The same flip of fnFL2h91v1 polled after the release and granted
at 2879 MiB (``waited=190 ms``). The fix: a complete leg whose staging is
still booked on the counter is not "will release nothing further" -- wait
(bounded) for exactly that staging, and only while it would cover the tag.
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
EPOCH = "1790432215.1"
STAGE = 617 * MIB            # one c2 staging (646533120 B, rounded)
FLOOR = 701 * MIB            # WEG2-CREDIT-FLOOR ... floor_mib=701 (MEASURED-D)
NEED = 1000 * MIB            # weights_8 on D TP2


def _bb1_counter(tmp_path, name, *, stagings: int = 2):
    """The card-2 book at TP2's weights_8 wait: published 9266, consumed 11127
    (the xsn290 overdraws of weights_2..7), PP2's c2 stagings booked and live,
    and the leg marked complete -- the state the refusing pass read."""
    c = VramCredit(name, credit_dir=str(tmp_path))
    c.begin_leg(EPOCH)
    c.publish("weights_family", 9266 * MIB)
    c.claim("weights_2..7", 11127 * MIB, epoch=EPOCH, overdraw=True)
    for i in range(stagings):
        # the counter is overdrawn, so the staging books as an OVERDRAW staging
        assert c.debit(f"c2-s{i}", STAGE, free_bytes=20000 * MIB, floor_bytes=FLOOR)
        c.mark_live(STAGE)
    c.leg_complete()
    st = c.read()
    assert st["leg_complete"] is True
    assert st["credit_bytes"] < st["consumed_bytes"] + NEED    # published < consumed + requested
    return c


def _wait(c, free, *, budget_s=5.0, alloc_poll_s=30.0):
    return c.wait_for(NEED, budget_s=budget_s, tag="weights_8",
                      free_bytes_now=1049 * MIB, free_reader=lambda: free["v"],
                      floor_bytes=FLOOR, epoch=EPOCH, poll_s=0.01,
                      alloc_poll_s=alloc_poll_s)


def test_bb1_complete_leg_with_booked_staging_is_waited_for_not_refused(tmp_path):
    """RED on e17bd548b5 (W35 on the first pass), GREEN with H111d."""
    c = _bb1_counter(tmp_path, "GPU-h111d-a")
    free = {"v": 1699 * MIB}                 # free_mib_at_refusal=1699

    def _pp2_leg_end():
        time.sleep(0.2)
        free["v"] = (1699 + 2 * 617) * MIB   # release_stage_buffers: raw_free ...
        for i in range(2):
            c.refund(f"c2-s{i}", STAGE, live_bytes=STAGE)   # ... then the refund

    th = threading.Thread(target=_pp2_leg_end)
    th.start()
    t0 = time.perf_counter()
    try:
        rec = _wait(c, free)
    finally:
        th.join()
    waited = time.perf_counter() - t0
    assert 0.15 <= waited < 3.0
    assert rec["free_bytes"] >= NEED + FLOOR
    assert c.read()["staged_bytes"] == 0


def test_complete_leg_with_card_already_holding_the_bytes_grants(tmp_path):
    """The order's literal mock (leg complete, published < consumed + requested,
    card free >= requested + floor): no W35. Already true on e17bd548b5 (#1349
    'ask the card once more') -- kept as the guard that H111d did not move it."""
    c = _bb1_counter(tmp_path, "GPU-h111d-b", stagings=0)
    free = {"v": (1000 + 701 + 5) * MIB}
    rec = c.wait_for(NEED, budget_s=5.0, tag="weights_8", free_bytes_now=1049 * MIB,
                     free_reader=lambda: free["v"], floor_bytes=FLOOR, epoch=EPOCH,
                     poll_s=0.01)
    assert rec["free_bytes"] == free["v"]


def test_complete_leg_no_staging_short_card_still_refuses_at_once(tmp_path):
    """Nothing booked, card short (1699 - 701 < 1000): the named refusal, now,
    exactly as before -- no new wait."""
    c = _bb1_counter(tmp_path, "GPU-h111d-c", stagings=0)
    free = {"v": 1699 * MIB}
    t0 = time.perf_counter()
    with pytest.raises(Weg2VramCreditRefused) as ei:
        _wait(c, free)
    assert time.perf_counter() - t0 < 0.5
    msg = str(ei.value)
    assert "W35" in msg and "peer_leg_complete=True" in msg


def test_complete_leg_whose_staging_would_not_cover_refuses_at_once(tmp_path):
    """Staging booked (1234 MiB) but free 300 - floor 701 + 1234 = 833 < 1000:
    even the staging does not fund the tag -> W35 at once, no wait."""
    c = _bb1_counter(tmp_path, "GPU-h111d-d")
    free = {"v": 300 * MIB}
    t0 = time.perf_counter()
    with pytest.raises(Weg2VramCreditRefused) as ei:
        _wait(c, free)
    assert time.perf_counter() - t0 < 0.5
    assert "peer_staged_mib=1234" in str(ei.value)


def test_booked_staging_never_refunded_refuses_after_the_bound(tmp_path):
    """No endless wait: a staging that never comes back is a W35 after
    ``alloc_poll_s``, named as such."""
    c = _bb1_counter(tmp_path, "GPU-h111d-e")
    free = {"v": 1699 * MIB}
    t0 = time.perf_counter()
    with pytest.raises(Weg2VramCreditRefused) as ei:
        _wait(c, free, budget_s=5.0, alloc_poll_s=0.3)
    waited = time.perf_counter() - t0
    assert 0.25 <= waited < 2.0
    assert "not refunded within" in str(ei.value)


def test_refunded_staging_with_card_still_short_refuses(tmp_path):
    """The staging came back but the card did not rise (someone else took it):
    the wait ends with the refund, W35 names it."""
    c = _bb1_counter(tmp_path, "GPU-h111d-f")
    free = {"v": 1699 * MIB}

    def _refund_only():
        time.sleep(0.15)
        for i in range(2):
            c.refund(f"c2-s{i}", STAGE, live_bytes=STAGE)

    th = threading.Thread(target=_refund_only)
    th.start()
    t0 = time.perf_counter()
    try:
        with pytest.raises(Weg2VramCreditRefused) as ei:
            _wait(c, free)
    finally:
        th.join()
    assert time.perf_counter() - t0 < 2.0
    assert "was refunded and the card is still short" in str(ei.value)
