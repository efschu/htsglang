"""SHORT-FLIP-REPRICE-1024 (SOLL R1 R2 R7 R11; Auftrag #220): a kurz request was classed gross after a flip and P
prefilled 158 tokens for it.

THE CASE (27B int8 rc41 1ecda02a58, boot boot_weg2_dkr27bbar1drei10241106_1ecda02a58_1024_110644, rid pdflip-21-215,
monitor 11:40:54Z ``SOLL-VERSTOSS R7 P-PREFILL-UNTER-X``):

  front 11:40:19.561  PDFLIP D-PROBE rid=pdflip-21-215 n=70999 device=53750 host=0 store=70134 tier=l2_arena
                      covered=70134 uncached_d=865 ... class_u=865
  front 11:40:19.564  WEG2 DECISION ... action=CLASSIFY rule=R2 ... S=865 ... X=12288 halfX=6144 classes=kurz
  front 11:40:19.564  PDFLIP ARRIVAL-SEAT rid=pdflip-21-215 verdict=d_prefill SHORT-IN-FLIP uncached=865 X=12288 epoch=21
  front 11:40:19.564  PDFLIP-ROUTE rid=pdflip-21-215 BATCH queued (awake=P admit_d=True ... remainder=865 queue=0 ...)
  D     11:40:20 TP0  MATCH-CENSUS-DEEP rid=pdflip-21-215 reached=53750 accepted=53750
  D     11:40:21 TP0  PDFLIP-CLAIM-SHORT rid=pdflip-21-215 pp=0 asked=17244 kv=17091 claimed=17091 lost=0 short=153
                      by=kv_prefix
  D     11:40:22 TP0  HiCache prefetch INCOMPLETE req=pdflip-21-215 completed_local=17091 completed_synced=17091 ...
                      matched=17091 loaded=0 ... deliverable=17244 shortfall=153
  D     11:40:22 TP0  PDFLIP W50-REROUTE rid=pdflip-21-215 d_extent=53908 X=12288 reason=store_short_after_flip
  D     11:40:22 TP0  PDFLIP D-NORECOMPUTE rid=pdflip-21-215 would_compute=53908 delivered=None -> RESUME-VIA-P
  front 11:40:22.576  RESUME-VIA-P CLASS rid=pdflip-21-215 u=53908 u_src=d_extent class=gross X=12288 d_extent=53908
  front 11:40:22.577  PDFLIP SET-VERDICT epoch=22 set=0 n=1 members=pdflip-21-215:53908 S=53908 ... verdict=COLLECT R4
  front 11:40:27.341  PDFLIP SET-REPROBE rid=pdflip-21-215 site=set-verdict u 865 -> 158 ... store=70841
  front 11:41:07.492  PDFLIP-USAGE-DETAILS rid=pdflip-21-215 ... p_computed=158 d_computed=5

THE ROOT (D): the store-short witness ``_pdflip_store_delivered`` was written SPAN-RELATIVE (the read [53750, 70994)
delivered 17091) and read as an absolute depth: 70999 - 17091 = 53908 instead of 70999 - (53750 + 17091) = 158.
THE GUARD (front): a d_extent that contradicts the kurz arrival class of a request that never ran is no class number.
Question (1): the ``BATCH queued`` line is written after every placement branch -- the entry stood once, in D's line.

Mutants are pinned in-process (``test_mutant_*``): with the fix switched off the metal numbers reappear.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_pdflip_store_short_fallback_rc12y as rc12y  # noqa: E402

from flliper.srt.managers import scheduler as sched_mod  # noqa: E402
from flliper.srt.mem_cache.hicache_storage import PrefetchOutcome  # noqa: E402
from flliper.srt.pdflip import d_match_probe as dmp  # noqa: E402
from flliper.srt.pdflip import d_norecompute as dn  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip import phase_policy as pp  # noqa: E402
from flliper.srt.pdflip import resume_via_p as rvp  # noqa: E402
from flliper.srt.pdflip import settle_writer as sw  # noqa: E402
from flliper.srt.pdflip import wake_cohort as wc  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="stage-a-pdflip-unit")

#: pdflip-21-215, the metal numbers
N, BASE, ASKED, GOT, X, HALFX = 70999, 53750, 17244, 17091, 12288, 6144
RID = "pdflip-16-42"   # the rc12y harness's rid


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_STORE_SHORT_TAIL", "0")
    monkeypatch.delenv(rc12y.MAX_CYCLES_ENV, raising=False)


def _arm(record, base=BASE, n=N):
    """The REAL ``Scheduler._pdflip_note_store_shortfall`` on the rc12y stand-in with ``record`` as the drain sees it."""
    s, r = rc12y._sched(n, 0, 1, X)
    s.tree_cache.prefetch_loaded_tokens_by_reqid[RID] = record
    if base is not None:
        r._prefetch_registered_prefix_len = base
    return s, r, s._pdflip_note_store_shortfall(r)


def _rec215():
    # HiCache prefetch INCOMPLETE ... completed_synced=17091 matched=17091 loaded=0 deliverable=17244
    return PrefetchOutcome(0, matched=GOT, deliverable=ASKED, synced=GOT)


def _dsched():
    return types.SimpleNamespace(server_args=types.SimpleNamespace(tp_prefill_max_tokens=X), page_size=64,
                                 ps=types.SimpleNamespace(tp_rank=0), pdflip_d_parked=[])


# ------------------------------------------------------------------------------------------------ D: the witness

def test_215_the_witness_is_the_absolute_depth_and_d_extends_the_158_token_tail():
    s, r, out = _arm(_rec215())
    assert out == "deferred"                       # the deferral itself is unchanged
    assert r._pdflip_store_delivered == BASE + GOT == 70841
    assert r._pdflip_store_deliverable == BASE + ASKED == 70994
    assert dn.d_would_compute(r) == N - 70841 == 158
    # FRONT-X 11:40:19.597: front_x=12288 front_priced=865 rode with the D-direct leg (no output yet)
    r.weg2_front_x, r.weg2_front_priced, r.output_ids = X, 865, []
    assert dn.d_direct_extends(_dsched(), r, dn.d_would_compute(r)) is True   # D's own extend: no needs-p, no flip
    # the dashboard sums stay store tokens (the span start is device-resident, not read)
    assert s._pdflip_store_short_delivered == GOT and s._pdflip_store_short_deliverable == ASKED


def test_mutant_without_the_span_start_the_metal_53908_reappears(monkeypatch):
    """Red-first, pinned: the pre-fix reading (span start 0) reproduces W50-REROUTE d_extent=53908 and the refusal."""
    monkeypatch.setattr(dn, "span_base", lambda req, record: 0)
    _s, r, _out = _arm(_rec215())
    assert dn.d_would_compute(r) == 53908
    r.weg2_front_x, r.weg2_front_priced, r.output_ids = X, 865, []
    assert dn.d_direct_extends(_dsched(), r, 53908) is False


def test_a_read_from_depth_0_is_byte_identical_the_parked_decode_family():
    """pdflip-8-58 (same boot, D 11:16:46): a parked decode resumed after the flip read its whole context from 0 --
    deliverable 57856 of 57928, the Mamba anchor cut the claim at 37852: the remainder 20076 is real (P computed 20076)."""
    rec = PrefetchOutcome(19564, matched=18288, deliverable=57856, synced=37852)
    _s, r, _out = _arm(rec, base=0, n=57928)
    assert r._pdflip_store_delivered == 37852 and dn.d_would_compute(r) == 20076
    _s, r2, _out = _arm(rec, base=None, n=57928)             # no stamp at all: the same
    assert r2._pdflip_store_delivered == 37852


def test_an_end_vote_record_keeps_its_reduced_end():
    rec = _rec215()
    rec.synced_end, rec.deliverable_end = 70841, 70994       # Form A: absolute already (HFB)
    _s, r, _out = _arm(rec)
    assert r._pdflip_store_delivered == 70841 and r._pdflip_store_deliverable == 70994


def test_a_stamp_that_cannot_be_this_reads_start_is_not_used():
    """base + deliverable beyond the request's ids: not this read's start -- the old reading, never an invented depth."""
    _s, r, _out = _arm(_rec215(), base=60000)
    assert r._pdflip_store_delivered == GOT
    assert dn.span_base(types.SimpleNamespace(_prefetch_registered_prefix_len=BASE), 17091) == 0   # bare int record


def test_refresh_after_a_whole_reread_follows_the_absolute_depth():
    """TAIL-CONSUME-ON-FULL companion: a re-read that completed must raise the witness to the depth it reached."""
    req = types.SimpleNamespace(rid="pdflip-21-215", full_untruncated_fill_ids=list(range(N)),
                                _prefetch_registered_prefix_len=BASE, _pdflip_store_delivered=BASE + GOT)
    whole = PrefetchOutcome(ASKED - GOT, matched=GOT, deliverable=ASKED, synced=ASKED)
    assert not whole.is_incomplete
    assert dn.refresh_delivered(req, whole) is True and req._pdflip_store_delivered == BASE + ASKED


def test_mutant_refresh_without_the_span_start_never_raises(monkeypatch):
    monkeypatch.setattr(dn, "span_base", lambda req, record: 0)
    req = types.SimpleNamespace(rid="pdflip-21-215", full_untruncated_fill_ids=list(range(N)),
                                _prefetch_registered_prefix_len=BASE, _pdflip_store_delivered=BASE + GOT)
    assert dn.refresh_delivered(req, PrefetchOutcome(ASKED - GOT, matched=GOT, deliverable=ASKED, synced=ASKED)) is False


def test_the_settle_remainder_and_the_cohort_price_read_the_depth():
    req = types.SimpleNamespace(rid="r", full_untruncated_fill_ids=list(range(N)), _prefetch_span_tokens=ASKED,
                                _prefetch_registered_prefix_len=BASE, origin_input_ids=list(range(N)), output_ids=[])
    rec = _rec215()
    assert sw.remainder(req, {"r": rec}) == 158
    assert wc.new_tokens(req, {"r": rec}) == 158
    # a bare int record (an admission site) keeps the old reading
    assert sw.remainder(req, {"r": 17091}) == N - 17091


# ------------------------------------------------------------------------------------------------ front: the guard

def _price(n, covered):
    return dmp.ExactPrice(uncached=n - covered, uncached_p=n - covered, covered=covered, tree_tokens=covered,
                          store_tokens=covered, image_in_tail=False, src="d_probe_store")


@pytest.fixture
def d_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.delenv(rvp.ENV, raising=False)
    return tmp_path


def _front():
    from test_ez_s1_review_1019 import _fresh
    f = _fresh()
    f.tag = "dkrtest"
    f.tp_prefill_max_tokens = X
    return f


def _take(f):
    async def body():
        return f._rvp_take()
    return asyncio.run(body())


def test_215_the_front_keeps_the_arrival_kurz_no_set_member_no_flip_reason(d_env, caplog):
    f = _front()
    f.__dict__.setdefault("_probe_price", {})["pdflip-21-215"] = _price(N, 70134)       # D-PROBE u=865
    rvp.write_request("pdflip-21-215", list(range(N)), 53908, X, dn.REASON)
    with caplog.at_level(logging.INFO):
        assert _take(f) == 1
    p = f.queue[-1]
    assert p.resume_via_p and p.p_only is False and p.est_uncached == 865
    assert pp.classify_arrival(p.est_uncached, X, 0.5, False) == pp.CLASS_KURZ
    assert F.Front._set_cands(f, [p], f._set_st()) == []          # R2: never a set member
    assert pp.immediate_park_trigger([p], X) is None              # R7: no flip reason
    assert f.counters["rvp_unrun_kurz_kept"] == 1
    assert any("RESUME-VIA-P CLASS rid=pdflip-21-215 u=865 u_src=d_probe_arrival class=kurz" in m for m in caplog.messages)
    assert any("RESUME-VIA-P KURZ-KEPT rid=pdflip-21-215" in m for m in caplog.messages)


def test_mutant_without_the_guard_the_front_classes_215_gross(d_env, monkeypatch):
    monkeypatch.setattr(F.Front, "_rvp_unrun_kurz_u", lambda self, rid, rec, d_ext: None)
    f = _front()
    f.__dict__.setdefault("_probe_price", {})["pdflip-21-215"] = _price(N, 70134)
    rvp.write_request("pdflip-21-215", list(range(N)), 53908, X, dn.REASON)
    assert _take(f) == 1
    p = f.queue[-1]
    assert p.est_uncached == 53908 and pp.immediate_park_trigger([p], X) is p      # the metal COLLECT R4


def test_a_parked_decode_keeps_ds_real_extent(d_env):
    """pdflip-8-58: arrival u=372 on D's device copy, decoded 468 tokens, parked for a flip, the sleep dropped the copy,
    the store holds the Mamba anchor at 37852 only: d_extent 20076 is real (P computed 20076) -- gross as before."""
    f = _front()
    f.__dict__.setdefault("_probe_price", {})["pdflip-8-58"] = _price(57460, 57088)
    rvp.write_request("pdflip-8-58", list(range(57928)), 20076, X, dn.REASON)
    assert _take(f) == 1
    assert f.queue[-1].est_uncached == 20076 and f.counters.get("rvp_unrun_kurz_kept", 0) == 0


@pytest.mark.parametrize("covered,d_ext,reason", [
    (N - 7000, 53908, dn.REASON),            # halb at the arrival: not kurz, D's extent stands
    (70134, 5000, dn.REASON),                # d_extent kurz itself: nothing to override
    (70134, 53908, "x_refusal_midstream"),   # another refusal: D's X-gate extent stands
])
def test_the_guard_is_narrow(d_env, covered, d_ext, reason):
    f = _front()
    f.__dict__.setdefault("_probe_price", {})["r1"] = _price(N, covered)
    rvp.write_request("r1", list(range(N)), d_ext, X, reason)
    assert _take(f) == 1
    assert f.queue[-1].est_uncached == d_ext and f.counters.get("rvp_unrun_kurz_kept", 0) == 0


def test_no_arrival_price_keeps_ds_extent(d_env):
    f = _front()
    rvp.write_request("r2", list(range(N)), 53908, X, dn.REASON)
    assert _take(f) == 1
    assert f.queue[-1].est_uncached == 53908


# ------------------------------------------------------------------------------------------------ question (1)

def test_the_batch_queued_line_names_where_the_entry_stands():
    """215: ``SHORT-IN-FLIP`` put the entry into D's admission line; the unconditional ``BATCH queued`` line (queue=0)
    read as a second, P-side placement. It names the placement now."""
    f = _front()
    p = F.Pending("pdflip-21-215", "/v1/messages", {}, "t", time.time(), asyncio.new_event_loop().create_future(),
                  est_prompt=N, est_uncached=865)
    f._ready_for_d = [p]
    f.queue = []
    assert F.Front._arrival_placed(f, p) == "d_line"
    f._ready_for_d, f.queue = [], [p]
    assert F.Front._arrival_placed(f, p) == "queue"
    f.queue = []
    assert F.Front._arrival_placed(f, p) == "none"
