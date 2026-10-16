"""STORE-HELD-DECIDE-1014 (NF h16 a2cfc77f76, boot ..._1014_151524, D TP0, pdflip-8-68).

pdflip-8-68 (5 images, 176133 tokens, lane 1) was routed SHORT -> D on credit=175744 src=d_served_anchor. D's
store read:

  15:27:50 #257 BELOW-ANCHOR read=9600  anchored=8320   #1324c store_hit=175872 delivered=8320
  15:27:55 #257 BELOW-ANCHOR read=65536 anchored=64576  #1324c store_hit=175872 delivered=64576
  15:28:00 / :06 / :11  identical: matched=64576 loaded=0, store_hit=175872, each re-read 5.2 s (l3fill 730 pages)
  15:28:16 fallback -> W31/W50 -> PARK-IMMEDIATE -> flip D->P

Every '#1068 PREFETCH DEFERRED reason=store_prefix_short attempt=1' says "the store did not hold the requested
prefix yet ... P's write-through is asynchronous" -- but the store HELD 175872 (P loaded all of it at 15:28:22-24,
'#988 LOADBACK ... anchor_depth=175872'). The wait exists for a writer; a read that ended short of what the store
already holds and did not grow on a re-read has none. 21 s of a stalled D -> one re-read (~5 s).
"""

import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_pdflip_store_short_fallback_rc12y import _cycle, _sched  # noqa: E402

from flliper.srt.managers import scheduler as sched_mod  # noqa: E402
from flliper.srt.mem_cache.hicache_storage import PrefetchOutcome  # noqa: E402

#: pdflip-8-68: (N, delivered after the 2nd read, store_hit, X)
H16 = (176133, 64576, 175872, 7005)
ENV = "FLLIPER_PDFLIP_STORE_SHORT_HELD_DECIDE"


@pytest.fixture(autouse=True)
def _nf(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_STORE_SHORT_TAIL", "0")
    monkeypatch.delenv("FLLIPER_PDFLIP_STORE_SHORT_MAX_CYCLES", raising=False)
    monkeypatch.delenv(ENV, raising=False)


def test_default_on():
    assert sched_mod.STORE_HELD_DECIDE_ENV == ENV
    r = type("R", (), {"_pdflip_store_delivered": 64576, "_pdflip_store_deliverable": 175872})()
    assert sched_mod._pdflip_store_held_short(r)


def test_pdflip_8_68_the_first_reread_without_growth_decides(caplog):
    n, delivered, held, x = H16
    s, r = _sched(n, delivered, held, x)
    with caplog.at_level(logging.WARNING, logger=sched_mod.logger.name):
        outs = [_cycle(s, r) for _ in range(2)]
    assert outs == ["deferred", "expired"], "metal: 4 identical re-reads, 21 s"
    text = caplog.text
    assert "W88-REROUTE" in text and "cause=over_x" in text and "store_held=175872" in text
    assert r in s.waiting_queue and r.prefetch_deferred is None


def test_a_remainder_within_x_is_released_to_d_after_one_reread(caplog):
    s, r = _sched(46850, 46656, 46848, 12288)  # rc12y numbers: the store held 192 more than the read gave
    with caplog.at_level(logging.WARNING, logger=sched_mod.logger.name):
        outs = [_cycle(s, r) for _ in range(2)]
    assert outs == ["deferred", "expired"]
    fb = [m for m in caplog.messages if m.startswith("PREFETCH-DEFER-FALLBACK")]
    assert len(fb) == 1 and "reason=store_held_no_growth" in fb[0]


def test_a_read_that_still_grows_is_waited_for():
    n, _delivered, held, x = H16
    s, r = _sched(n, 8320, held, x)
    assert _cycle(s, r) == "deferred"
    s.tree_cache.prefetch_loaded_tokens_by_reqid["pdflip-16-42"] = PrefetchOutcome(
        64576, matched=0, deliverable=held, synced=64576)
    assert _cycle(s, r) == "deferred", "8320 -> 64576 grew: a re-read is owed"


def test_a_sleeping_d_still_does_not_count():
    n, delivered, held, x = H16
    s, r = _sched(n, delivered, held, x)
    s.pdflip_dormant = True
    assert [_cycle(s, r) for _ in range(6)] == ["deferred"] * 6  # P may still publish (xsn344)


def test_off_is_the_cycle_bound(monkeypatch):
    monkeypatch.setenv(ENV, "0")
    n, delivered, held, x = H16
    s, r = _sched(n, delivered, held, x)
    outs = [_cycle(s, r) for _ in range(5)]
    assert outs == ["deferred"] * 4 + ["expired"]
