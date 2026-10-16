"""TAIL-CONSUME-ON-FULL (NF h16h 853534a9ff, boot _211215, D 00:49:44-45,
weg2-182-1661 -- the h48 pendulum, one request three times over P, 106 s).

THE METAL (D.log):
* Z.900990 TP2 ``READ-STAGES pages=810`` of 1417 (the arena reap race,
  Z.900987 ``LATE-COMPLETE n=14 statuses=[3]`` the same second); the group END
  ``completed_synced=51840`` on all three ranks (Z.901047/901049/901053 --
  TP0/TP1 ``completed_local=90688``, TP2 51840: the Form A MIN).
* the FIRST (short) reap's ``tail_adopt.agree`` ran the H63b receipt:
  Z.901059/901060 ``WEG2-TAIL-CONSUMED removed_files=4/2``;
* the #1456 HOLD-REFETCH read every page (Z.901154 ``completed_synced=90688``),
  its re-staging found nothing: Z.901147 ``TAIL-READY verdict=no_parts``;
* Z.901158 ``#x38 SETTLE-TAIL tail=12 ... record-short`` -> ``D-NORECOMPUTE
  would_compute=38861`` (= 90701 - 51840: the witness of the FIRST read) ->
  ``W50-REROUTE`` -> P prefilled 90701 tokens again.

THE FIX: (a) the receipt only at a FULL reap, judged on the group-synced END
and hit (``tail_adopt.reap_full``); a short reap defers it (named) until a
full reap, the admission, an abort, or the cap; (b) the release refreshes the
stale witness from the record's group-agreed completion before
``d_would_compute``. Rig: the H63b three-rank D rig (real tail_handoff files,
real staging / vote / agree / plan_adopt)."""
from __future__ import annotations

import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.mem_cache.hicache_storage import PrefetchOutcome  # noqa: E402
from sglang.srt.weg2 import d_norecompute as dnr  # noqa: E402
from sglang.srt.weg2 import tail_adopt as ta  # noqa: E402

from test_weg2_tail_keep_h63b import (  # noqa: E402,F401
    N, PREFIX, RID, _join, _publish_parts, _req, _rids_on_disk, _stage_and_vote, d_group,
)

#: the metal's ranks: completed_local per TP rank, the group END, the store hit (1417 pages)
LOCAL = [90688, 90688, 51840]
SYNCED, HIT = 51840, 90688


def _has_fix() -> bool:
    return "full" in inspect.signature(ta.agree).parameters


def _agree(rank, vote: int, full: bool) -> None:
    with rank.active():
        if _has_fix():
            ta.agree(RID, vote, full=full, completed_synced=SYNCED if not full else HIT, hit_synced=HIT)
        else:
            ta.agree(RID, vote)              # the base: the receipt at every reap
    _join("weg2-tail-consumed")


def _short_then_refetch(ranks, d):
    """The h48 sequence on three D ranks: staging, a SHORT first reap (the group
    vote 0: one rank's staging still running at the short reap -- no usable
    agreed entry), then the HOLD-REFETCH: re-stage, full reap."""
    _publish_parts()
    _stage_and_vote(ranks)
    for r in ranks:
        _agree(r, 0, full=False)
    on_disk_after_short = _rids_on_disk(d)
    votes2 = _stage_and_vote(ranks)
    for r in ranks:
        _agree(r, min(votes2), full=True)
    return on_disk_after_short, votes2


def _would_compute(rank):
    """The settle release's D-NORECOMPUTE question after the full re-read: the
    short read's stale witness, the full record (group-agreed)."""
    req = _req()
    req._weg2_store_delivered = PREFIX - 64            # the FIRST read's witness (stale)
    rec = PrefetchOutcome(PREFIX, hit_tokens=PREFIX, probed=True, matched=0,
                          deliverable=PREFIX, synced=PREFIX)
    with rank.active():
        if hasattr(dnr, "refresh_delivered"):
            dnr.refresh_delivered(req, rec)
        return dnr.d_would_compute(req), req


def test_h48_a_short_reap_keeps_the_parts_and_the_refetch_adopts(d_group):
    ranks, d = d_group
    with envs.SGLANG_WEG2_TAIL_KEEP_MIB.override(512):
        on_disk, votes2 = _short_then_refetch(ranks, d)
        assert on_disk == [RID], "base: WEG2-TAIL-CONSUMED at the short reap"
        assert votes2 == [2, 2, 2], "base: the refetch's re-staging finds no parts (no_parts -> vote 0)"
        rems = [_would_compute(r)[0] for r in ranks]
        assert rems == [0, 0, 0], "no W50-REROUTE: the agreed E2 window adopts the tail"
        plans = []
        for r in ranks:
            req = _req()
            with r.active():
                plans.append(ta.plan_adopt(req, len(req.prefix_indices), batch_empty=True))
        _join("weg2-tail-consumed")
    assert all(p is not None and p.skip for p in plans)
    assert _rids_on_disk(d) == [], "the full reap's receipt removed them"


def test_mutant_the_receipt_at_the_short_reap_is_the_metal(d_group, monkeypatch):
    """MUTANT: consumed at the short reap (the switch off = the base form)."""
    ranks, d = d_group
    monkeypatch.setenv(ta.CONSUME_ON_FULL_ENV, "0")
    with envs.SGLANG_WEG2_TAIL_KEEP_MIB.override(512):
        on_disk, votes2 = _short_then_refetch(ranks, d)
        rems = [_would_compute(r)[0] for r in ranks]
    assert on_disk == [] and votes2 == [0, 0, 0]
    assert all(x > 0 for x in rems), "would_compute > 0 -> RESUME-VIA-P (W50-REROUTE)"


def test_reap_full_reads_the_group_synced_values_on_every_rank():
    """Rank agreement (Form A MIN): the metal's completed_local differ per rank
    (90688 / 90688 / 51840), the reduced END is 51840 on all three; the verdict
    from the synced values is one value, short, on every rank."""
    assert [ta.reap_full(SYNCED, HIT, True) for _ in LOCAL] == [False, False, False]
    assert ta.reap_full(HIT, HIT, True) is True
    assert ta.reap_full(0, 0, False) is True               # unprobed: the receipt as before


def test_mutant_rank_local_values_split_the_group():
    """MUTANT 2: the verdict on each rank's own completed count -- TP0/TP1 would
    remove the shared files TP2's refetch still has to stage."""
    assert [ta.reap_full(c, HIT, True) for c in LOCAL] == [True, True, False]


def test_the_reap_passes_the_reduced_values():
    """Pin: the reap site feeds reap_full the reduced END and hit, never the
    rank-local completed_tokens / _hit_tokens_local."""
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    src = inspect.getsource(UnifiedRadixCache.check_prefetch_progress)
    # H16N-REVIEW 1015: the RAW reduced ENDS (bit-identical on every rank), not the
    # per-rank unpacked counts (test_weg2_h16n_review_1015)
    assert "reap_full(int(_synced_end), int(_hit_end), bool(_probed))" in src
    i = src.index("reap_full(")
    assert "_hit_tokens_local" not in src[i:i + 200]
    assert src.index("_reap_annotation_from_packed(packed)") < i


def _deferred_one(ranks, d):
    _publish_parts()
    _stage_and_vote(ranks)
    for r in ranks:
        _agree(r, 0, full=False)
    assert _rids_on_disk(d) == [RID]


def test_a_deferred_receipt_is_released_at_an_abort(d_group):
    ranks, d = d_group
    with envs.SGLANG_WEG2_TAIL_KEEP_MIB.override(512):
        _deferred_one(ranks, d)
        with ranks[0].active():
            ta.drop(RID)
        _join("weg2-tail-consumed")
    assert _rids_on_disk(d) == []


def test_a_deferred_receipt_is_released_at_the_admission(d_group):
    ranks, d = d_group
    with envs.SGLANG_WEG2_TAIL_KEEP_MIB.override(512):
        _deferred_one(ranks, d)
        req = _req()
        with ranks[0].active():
            ta.plan_adopt(req, len(req.prefix_indices), batch_empty=True)
        _join("weg2-tail-consumed")
    assert _rids_on_disk(d) == []


def test_the_deferred_set_is_bounded(monkeypatch):
    released = []
    monkeypatch.setattr(ta.th, "consumed", lambda rid: released.append(rid) or True)
    monkeypatch.setattr(ta, "_CONSUME_DEFERRED", {})
    for i in range(ta.CONSUME_DEFERRED_CAP + 3):
        ta._defer_consume(f"r{i}", 1, 2, False)
    assert released == ["r0", "r1", "r2"]
    assert len(ta._CONSUME_DEFERRED) == ta.CONSUME_DEFERRED_CAP


def test_the_witness_follows_the_full_record_never_down():
    from types import SimpleNamespace

    req = SimpleNamespace(rid="x", _weg2_store_delivered=51840)
    full = PrefetchOutcome(90688, hit_tokens=90688, probed=True, matched=51840, deliverable=90688, synced=90688)
    full.synced_end = 90688
    assert dnr.refresh_delivered(req, full) and req._weg2_store_delivered == 90688
    short = PrefetchOutcome(51840, hit_tokens=90688, probed=True, matched=0, deliverable=90688, synced=51840)
    req2 = SimpleNamespace(rid="y", _weg2_store_delivered=51840)
    assert not dnr.refresh_delivered(req2, short) and req2._weg2_store_delivered == 51840
    low = PrefetchOutcome(3008, hit_tokens=3008, probed=True, matched=0, deliverable=3008, synced=3008)
    req3 = SimpleNamespace(rid="z", _weg2_store_delivered=51840)
    assert not dnr.refresh_delivered(req3, low) and req3._weg2_store_delivered == 51840
    req4 = SimpleNamespace(rid="w")                       # no short read on record: untouched
    assert not dnr.refresh_delivered(req4, full) and not hasattr(req4, "_weg2_store_delivered")
