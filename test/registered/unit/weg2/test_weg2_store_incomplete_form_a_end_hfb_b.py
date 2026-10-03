"""HFB-b: ``PrefetchOutcome.is_incomplete`` is a GROUP verdict under the
Form A END vote, also when a span base is not page-aligned.

HFB (38945b1880) made the store-short DELIVERED depth the group's (the reduced
END). The incompleteness itself still compared the span-relative pair
``synced < deliverable``: ``deliverable`` is ``len(prefetch_key)`` floored to
pages, and each rank's span starts at its OWN base. With one group END 29760
and page 64, a host base 12500 (not a page multiple) floors its 17260-token
span to 17216 (END 29716) while a worker base 16384 keeps 13376 (END 29760).
A reduced END of 29716 is then COMPLETE on the host and INCOMPLETE on the
workers -- the host admits, the workers defer: the same queue split that
killed rc12z21 (PrefetchBallotDigestMismatch), reached through the other
term. On the metal every base so far was a page multiple (12544, 15104,
16384), so this is the closing of the class, not a measured death.

GREEN with HFB-b: an END-vote record carries ``deliverable_end`` (the page
floor of the group END, P4b-capped on the reduced hit/synced ENDs) beside
``synced_end``, and ``is_incomplete`` compares those two. RED on 38945b1880:
the attributes are ignored and the ranks answer differently. A record
without an END vote reads exactly as before.
"""

import inspect

from sglang.srt.mem_cache import unified_radix_cache as urc
from sglang.srt.mem_cache.hicache_storage import PrefetchOutcome

PAGE = 64
GROUP_END = 29760
HOST_BASE = 12500            # not a page multiple
WORKER_BASE = 16384
REDUCED_END = 29716          # the host's own floored END


def _floor(n):
    return (n // PAGE) * PAGE


def _record(base, synced_end=REDUCED_END, no_writer=False, hit_end=0, end_vote=True):
    """The record check_prefetch_progress builds on one rank (span-relative
    fields exactly as before, plus the two ENDs under the END vote)."""
    span = GROUP_END - base
    deliverable = _floor(span)
    synced = max(0, synced_end - base)
    rec = PrefetchOutcome(0, matched=synced, deliverable=deliverable, synced=synced)
    if end_vote:
        rec.synced_end = synced_end
        rec.deliverable_end = urc._hfb_deliverable_end(
            base, span, no_writer, hit_end, synced_end, PAGE
        ) if hasattr(urc, "_hfb_deliverable_end") else _floor(GROUP_END)
    return rec


def _verdicts(**kw):
    return [_record(b, **kw).is_incomplete for b in (HOST_BASE, WORKER_BASE, WORKER_BASE)]


def test_unaligned_host_base_one_verdict_for_the_group():
    # span-relative: host 17216 of 17216 (complete), worker 13332 of 13376 (short)
    host, w1, w2 = _verdicts()
    assert host == w1 == w2, (
        f"host={host} workers={w1},{w2}: the host admits while the workers defer "
        "-- the rc12z21 queue split through is_incomplete"
    )
    assert host is True, "29716 < 29760: the group read is short by 44 tokens"


def test_a_complete_group_read_is_complete_everywhere():
    assert _verdicts(synced_end=GROUP_END) == [False, False, False]


def test_the_cap_uses_the_group_ends():
    # no writer can extend it and the store answered 29716: complete everywhere
    assert _verdicts(no_writer=True, hit_end=REDUCED_END) == [False, False, False]
    assert urc._hfb_deliverable_end(HOST_BASE, GROUP_END - HOST_BASE, True,
                                    REDUCED_END, REDUCED_END, PAGE) == 29696
    assert urc._hfb_deliverable_end(WORKER_BASE, GROUP_END - WORKER_BASE, False,
                                    0, REDUCED_END, PAGE) == GROUP_END


def test_without_an_end_vote_the_verdict_is_unchanged():
    rec = _record(0, synced_end=13000, end_vote=False)
    assert not hasattr(rec, "deliverable_end")
    assert rec.is_incomplete == (13000 < _floor(GROUP_END))


def test_the_tree_stamps_both_ends_under_the_end_vote():
    src = inspect.getsource(urc.UnifiedRadixCache.check_prefetch_progress)
    assert "_rec.deliverable_end = _hfb_deliverable_end(" in src
    assert "_hit_end = int(_hit_tokens)" in src
