"""HFB (rc12z21 D 13:35:14, rid pdflip-8-34): the store-short bound decides on
the GROUP's delivered depth, not on each rank's span-relative count.

MEASURED (D log 46240-46746). The read ran under the Form A END vote (HP1):
TP0's span started at the host base (12544, later 15104), the expert workers'
at their own 16384. Every ``#1324`` record therefore carried a DIFFERENT
``materialized`` per rank -- TP0 13376 on every pass, the workers 9536, 9536,
12096, 12096, 9536 -- while the reduced END was one number on all three ranks
(25920, 28480). ``_pdflip_note_store_shortfall`` stamped ``materialized`` as the
delivered prefix; the fresh-mark cycle bound saw growth on the workers (9536
-> 12096) and none on TP0, so only TP0 reached ``cycles=5 bound=4`` and took
``PREFETCH-DEFER-FALLBACK ... reason=over_x`` -> W88 503. TP0's queue emptied,
the workers kept the rid and voted ``HP1 FORM-A END-VOTE`` for it, and the
next ballot died ``PrefetchBallotDigestMismatch ... group_min=0
group_max=82296685 queue_len=0``.

Driven through the REAL ``Scheduler._pdflip_note_store_shortfall`` ->
``_apply_prefetch_deferral`` -> ``_pdflip_store_short_fallback`` chain on three
stand-in ranks (the rc12y harness), fed the metal records pass by pass. RED
on 82fa502795: the ranks answer differently on pass 5. GREEN with HFB: the
record carries the reduced END (``synced_end``) and every rank decides on it
-- the same answer on every pass, and the remainder priced from the absolute
depth. A record without an END vote (every other boot) is untouched.
"""

import inspect
import logging
import os
import sys

import pytest

# pdflip/ is a package: a bare sibling import only resolves when pytest runs
# from this directory. Put the directory on the path (the pattern of
# test_pdflip_settle_writer_veto_p4b_0928) so the repo root collects it too.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_pdflip_store_short_fallback_rc12y as rc12y  # noqa: E402  (the rc12y harness)

from flliper.srt.managers import scheduler as sched_mod
from flliper.srt.mem_cache import unified_radix_cache as urc
from flliper.srt.mem_cache.hicache_storage import PrefetchOutcome

RID = "pdflip-16-42"          # the harness's rid; the metal rid was pdflip-8-34
N = 29808                   # 13376 + tail 16432 (TP0's PREFETCH-DEFER-FALLBACK)
X = 12288
WORKER_BASE = 16384
#: (host base, TP0 materialized, worker materialized) per pass, rc12z21
PASSES = [
    (12544, 13376, 9536),
    (12544, 13376, 9536),
    (15104, 13376, 12096),
    (15104, 13376, 12096),
    (12544, 13376, 9536),
    (12544, 13376, 9536),
    (12544, 13376, 9536),
    (12544, 13376, 9536),
]


@pytest.fixture(autouse=True)
def _nf_profile(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_STORE_SHORT_TAIL", "0")
    monkeypatch.delenv(rc12y.MAX_CYCLES_ENV, raising=False)


def _record(base, got, host_base, end_vote=True):
    """The #1324 record as check_prefetch_progress writes it: span-relative
    counts, the span's page-floored length, and (END vote) the reduced END --
    the host's END, the group MIN (25920 / 28480 on the metal)."""
    end = host_base + 13376
    span = 29760 - base
    rec = PrefetchOutcome(0, matched=got, deliverable=span, synced=got)
    if end_vote:
        rec.synced_end = end
    return rec


def _ranks():
    return [rc12y._sched(N, 13376, 17216, X) for _ in range(3)]


def _pass(ranks, host_base, tp0_got, worker_got, end_vote=True):
    outs = []
    for i, (s, r) in enumerate(ranks):
        if r not in s.waiting_queue:
            outs.append("gone")
            continue
        base, got = (host_base, tp0_got) if i == 0 else (WORKER_BASE, worker_got)
        s.tree_cache.prefetch_loaded_tokens_by_reqid[RID] = _record(base, got, host_base, end_vote)
        outs.append(rc12y._cycle(s, r))
    return outs


def test_rc12z21_every_rank_takes_the_same_verdict_on_every_pass(caplog):
    ranks = _ranks()
    with caplog.at_level(logging.WARNING, logger=sched_mod.logger.name):
        seen = [_pass(ranks, *p) for p in PASSES]
    for k, outs in enumerate(seen, 1):
        assert len(set(outs)) == 1, (
            f"pass {k}: ranks answered {outs} -- rc12z21: TP0 W88 alone, the "
            "workers kept the rid (PrefetchBallotDigestMismatch)"
        )
    queues = [r in s.waiting_queue for s, r in ranks]
    assert len(set(queues)) == 1, f"queues diverged: {queues}"


def test_the_bound_counts_the_group_end_and_prices_the_absolute_remainder(caplog):
    ranks = _ranks()
    with caplog.at_level(logging.WARNING, logger=sched_mod.logger.name):
        seen = [_pass(ranks, *p) for p in PASSES]
    # growth 25920 -> 28480 on pass 3 restarts the count on EVERY rank; the
    # fall back to 25920 is no growth: passes 3..7 = cycles 1..5 -> pass 7
    assert [o[0] for o in seen[:6]] == ["deferred"] * 6
    assert seen[6] == ["expired"] * 3
    fb = [rec.getMessage() for rec in caplog.records
          if rec.getMessage().startswith("PREFETCH-DEFER-FALLBACK")]
    assert len(fb) == 3 and all("reason=no_writer_progress" in m for m in fb), fb
    # the remainder from the depth the group holds (29808 - 28480 best, 25920
    # delivered): within X -> released to admission, never the W88 503
    assert all(f"delivered=25920 tail={N - 25920}" in m for m in fb), fb
    assert "W88" not in caplog.text


def test_without_an_end_vote_the_record_reads_as_before(caplog):
    """No END vote (27B, even TP, classic D): no ``synced_end`` on the record,
    ``materialized`` stays the delivered term -- the pre-HFB answer."""
    s, r = rc12y._sched(N, 13376, 17216, X)
    s.tree_cache.prefetch_loaded_tokens_by_reqid[RID] = _record(12544, 13376, 12544, end_vote=False)
    assert rc12y._cycle(s, r) == "deferred"
    assert r._pdflip_store_delivered == 13376


def test_the_tree_stamps_the_reduced_end_only_under_the_end_vote():
    src = inspect.getsource(urc.UnifiedRadixCache.check_prefetch_progress)
    assert "_synced_end = int(packed[0].item())" in src
    assert "if _eb_rec is not None:" in src and ".synced_end = _synced_end" in src
