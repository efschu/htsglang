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


def _arm(record, base=BASE, n=N, stamp=None):
    """The REAL ``Scheduler._pdflip_note_store_shortfall`` on the rc12y stand-in with ``record`` as the drain sees it.
    ``base`` is the span start the record's OPERATION carried (review MUSS-1: ``record.span_start``); ``stamp`` the
    request's own ``_prefetch_registered_prefix_len`` (default: the same value, as at the first observation)."""
    s, r = rc12y._sched(n, 0, 1, X)
    if base is not None:
        record.span_start = base
    s.tree_cache.prefetch_loaded_tokens_by_reqid[RID] = record
    st = base if stamp is None else stamp
    if st is not None:
        r._prefetch_registered_prefix_len = st
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


def test_a_start_that_cannot_be_this_reads_is_not_used():
    """start + deliverable beyond the request's ids: not this read's start -- the old reading, never an invented depth."""
    _s, r, _out = _arm(_rec215(), base=60000)
    assert r._pdflip_store_delivered == GOT
    assert dn.span_base(types.SimpleNamespace(_prefetch_registered_prefix_len=BASE), 17091) == 0   # bare int record


def test_review_muss1_the_request_stamp_alone_is_never_a_base():
    """MUSS-1: a record without its operation's start keeps the span-relative reading, whatever the request carries."""
    _s, r, _out = _arm(_rec215(), base=None, stamp=BASE)
    assert r._pdflip_store_delivered == GOT


def test_review_muss1_rank_skew_two_stale_stamps_one_record_one_witness():
    """R11: two ranks re-stamped by a rank-local re-match (W65 skew) to different depths; the record (group-agreed
    span, its operation's start) is the same -> the witness is the same on both."""
    outs = []
    for stamp in (70841, 70266):
        _s, r, _out = _arm(_rec215(), stamp=BASE)
        r._prefetch_registered_prefix_len = stamp          # the next _prefetch_kvcache call on this rank
        _s._pdflip_note_store_shortfall(r)
        outs.append((r._pdflip_store_delivered, dn.d_would_compute(r)))
    assert outs == [(70841, 158), (70841, 158)]


def test_review_muss1_the_record_carries_its_operations_start():
    """MUSS-1 wiring: the start is stamped once on the operation at registration and copied at BOTH record writes."""
    import inspect

    from flliper.srt.mem_cache import unified_radix_cache as urc

    assert PrefetchOutcome(5).span_start is None
    src = inspect.getsource(urc.UnifiedRadixCache.prefetch_from_storage)
    assert "operation._pdflip_span_start = None if _ss is None else int(_ss)" in src
    assert "_ss = span_base" in src      # 27B: else key_base; NF: span_base only (the tree takes no key_base)
    full = inspect.getsource(urc)
    assert full.count('.span_start = getattr(operation, "_pdflip_span_start", None)') == 1      # termination
    assert full.count('"_pdflip_span_start", None)') == 2                                          # + revoke
    assert "_prefetch_registered_prefix_len" not in inspect.getsource(dn.span_base).split('"""')[-1]


def test_refresh_after_a_whole_reread_follows_the_absolute_depth():
    """TAIL-CONSUME-ON-FULL companion: a re-read that completed must raise the witness to the depth it reached."""
    req = types.SimpleNamespace(rid="pdflip-21-215", full_untruncated_fill_ids=list(range(N)),
                                _prefetch_registered_prefix_len=BASE, _pdflip_store_delivered=BASE + GOT)
    whole = PrefetchOutcome(ASKED - GOT, matched=GOT, deliverable=ASKED, synced=ASKED)
    whole.span_start = BASE
    assert not whole.is_incomplete
    assert dn.refresh_delivered(req, whole) is True and req._pdflip_store_delivered == BASE + ASKED


def test_mutant_refresh_without_the_span_start_never_raises(monkeypatch):
    monkeypatch.setattr(dn, "span_base", lambda req, record: 0)
    req = types.SimpleNamespace(rid="pdflip-21-215", full_untruncated_fill_ids=list(range(N)),
                                _prefetch_registered_prefix_len=BASE, _pdflip_store_delivered=BASE + GOT)
    whole = PrefetchOutcome(ASKED - GOT, matched=GOT, deliverable=ASKED, synced=ASKED)
    whole.span_start = BASE
    assert dn.refresh_delivered(req, whole) is False


def test_the_settle_remainder_and_the_cohort_price_read_the_depth():
    req = types.SimpleNamespace(rid="r", full_untruncated_fill_ids=list(range(N)), _prefetch_span_tokens=ASKED,
                                _prefetch_registered_prefix_len=BASE, origin_input_ids=list(range(N)), output_ids=[])
    rec = _rec215()
    rec.span_start = BASE
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
    f.__dict__.setdefault("_arrival_price", {})["pdflip-21-215"] = _price(N, 70134)       # D-PROBE u=865
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
    f.__dict__.setdefault("_arrival_price", {})["pdflip-21-215"] = _price(N, 70134)
    rvp.write_request("pdflip-21-215", list(range(N)), 53908, X, dn.REASON)
    assert _take(f) == 1
    p = f.queue[-1]
    assert p.est_uncached == 53908 and pp.immediate_park_trigger([p], X) is p      # the metal COLLECT R4


def test_a_parked_decode_keeps_ds_real_extent(d_env):
    """pdflip-8-58: arrival u=372 on D's device copy, decoded 468 tokens, parked for a flip, the sleep dropped the copy,
    the store holds the Mamba anchor at 37852 only: d_extent 20076 is real (P computed 20076) -- gross as before."""
    f = _front()
    f.__dict__.setdefault("_arrival_price", {})["pdflip-8-58"] = _price(57460, 57088)
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
    f.__dict__.setdefault("_arrival_price", {})["r1"] = _price(N, covered)
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


# ------------------------------------------------------------------------------------------------ review SOLL-2
# Three readers of the witness change behaviour whenever a read's span starts above 0 (named in the commit): each
# moves to its documented absolute intent. Pinned here on the 215 numbers, with the pre-fix reading as the mutant.

def _witnessed(mutant, monkeypatch):
    if mutant:
        monkeypatch.setattr(dn, "span_base", lambda req, record: 0)
    _s, r, _out = _arm(_rec215())
    return r


@pytest.mark.parametrize("mutant,expect", [(False, True), (True, False)])
def test_review_soll2_capacity_park_candidate_reads_the_absolute_deliverable(monkeypatch, mutant, expect):
    """#248h: the store holds [0, 70994) of 70999 -- a whole re-read is admitted (5 <= X): capacity park, no P leg.
    Span-relative (deliverable 17244) it read 53755 > X: not a candidate."""
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.delenv(rvp.ENV, raising=False)
    r = _witnessed(mutant, monkeypatch)
    r.stream, r.multimodal_inputs, r.priority, r._pdflip_store_short_fallback = True, None, 0, True
    assert rvp.capacity_park_candidate(r, X, now=0.0) is expect


@pytest.mark.parametrize("mutant,expect", [(False, True), (True, False)])
def test_review_soll2_p_span_delivered_reads_the_absolute_depth(monkeypatch, mutant, expect):
    """S75 P-span label: P wrote 70841 tokens (1106 pages); the read reached 70841 -> nothing left for P to write."""
    from flliper.srt.pdflip.handoff_keys import P_PAGES_ATTR, P_TOKENS_ATTR

    r = _witnessed(mutant, monkeypatch)
    setattr(r, P_PAGES_ATTR, 70841 // 64)
    setattr(r, P_TOKENS_ATTR, 70841)
    assert sw.p_span_delivered(r, 64) is expect


@pytest.mark.parametrize("mutant,expect", [(False, False), (True, True)])
def test_review_soll2_window_above_delivered_reads_the_absolute_depth(monkeypatch, mutant, expect):
    """ZR-2: an agreed END window at page 70784 lies BELOW the delivered 70841 -- no wait for a re-read.
    Span-relative (17091) the settle waited for a re-read that could add nothing."""
    from flliper.srt.pdflip import tail_adopt as ta

    r = _witnessed(mutant, monkeypatch)
    ta._AGREED[str(r.rid)] = types.SimpleNamespace(
        agreed=True, staged=types.SimpleNamespace(spec=types.SimpleNamespace(page_prefix=70784)))
    try:
        assert ta.window_above_delivered(r) is expect
    finally:
        ta._AGREED.pop(str(r.rid), None)


# ------------------------------------------------------------------------------------------------ review SOLL-3

def test_review_soll3_a_second_p_leg_keeps_ds_extent(d_env):
    """attempt=2: a P leg already ran for this rid -- D's extent after it is no arrival question."""
    f = _front()
    f.__dict__.setdefault("_arrival_price", {})["pdflip-21-215"] = _price(N, 70134)
    rvp.write_request("pdflip-21-215", list(range(N)), 53908, X, dn.REASON, attempt=2)
    assert _take(f) == 1
    assert f.queue[-1].est_uncached == 53908 and f.counters.get("rvp_unrun_kurz_kept", 0) == 0


def test_review_soll3_a_repriced_latest_price_is_not_the_arrival_price(d_env):
    """A decoded request: its arrival price is on the PROMPT (57460), a later ``_rvp_reprice`` put the LEG ids' price
    (57928, 372 uncached) into ``_probe_price``. Its next store_short leg (ids == the repriced ids) must not read as
    'never ran': only the write-once arrival book counts."""
    f = _front()
    f.__dict__.setdefault("_probe_price", {})["pdflip-8-58"] = _price(57928, 57556)          # latest, leg ids
    F.Front._note_arrival_price(f, "pdflip-8-58", _price(57460, 57088))                       # the arrival
    F.Front._note_arrival_price(f, "pdflip-8-58", _price(57928, 57556))                       # write-once: ignored
    assert f._arrival_price["pdflip-8-58"].covered == 57088
    rvp.write_request("pdflip-8-58", list(range(57928)), 20076, X, dn.REASON)
    assert _take(f) == 1
    assert f.queue[-1].est_uncached == 20076


def test_review_soll3_the_arrival_sites_write_the_arrival_book():
    import inspect

    src = inspect.getsource(F)
    assert src.count("Front._note_arrival_price(self, rid, _xp)") == 2     # arrival D-PROBE + its NF-1b late answer
    assert '(self.__dict__.get("_arrival_price") or {}).get(str(rid))' in inspect.getsource(F.Front._rvp_unrun_kurz_u)


# ------------------------------------------------------------------------------------------------ review SOLL-4

def _settle_sched(tmp_path, monkeypatch, record):
    """A bare Scheduler with the REAL settle tick / refetch / #1324 arm / D-NORECOMPUTE release; the store is the one
    record of 215's first read (its operation's start 53750); ``_prefetch_kvcache`` is the metal's DECLINED re-issue
    (D 11:41:02 ``#1456 HOLD-REFETCH ... declined:vote_negative``) that re-stamps the request from a fresh re-match
    and registers nothing -- the record stays."""
    S = sched_mod.Scheduler
    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    (tmp_path / "handoff").mkdir()
    base, r = rc12y._sched(N, 0, 1, X)
    s = S.__new__(S)
    for k, v in vars(base).items():
        if not callable(v) or k in ("tree_cache", "server_args", "ipc_channels"):
            setattr(s, k, v)
    s.ps = types.SimpleNamespace(tp_size=1, pp_size=1, tp_rank=0)
    s.page_size = 64
    s.pdflip_dormant_hold = []
    s._pdflip_wake_seq = 1
    s.pdflip_d_parked = []
    record.span_start = BASE
    s.tree_cache.prefetch_loaded_tokens_by_reqid = {r.rid: record}
    s.tree_cache.ongoing_prefetch = {}
    s.tree_cache.check_prefetch_progress = lambda rid: rid not in s.tree_cache.ongoing_prefetch
    s.tree_cache.cache_controller = types.SimpleNamespace(pdflip_hold_rids=set(), prefetch_rate_limited=lambda: False,
                                                          host_role="staging")
    calls = []

    def _declined(req, limit_tokens=None):
        req._prefetch_registered_prefix_len = BASE + GOT      # the re-match of the re-issue
        calls.append(req.rid)
        return "declined:vote_negative"

    s._prefetch_kvcache = _declined
    r._prefetch_registered_prefix_len = BASE
    r._prefetch_span_tokens = ASKED
    r.output_ids, r.stream, r.multimodal_inputs = [], True, None
    r.weg2_front_x, r.weg2_front_priced = X, 865             # FRONT-X 11:40:19.597
    r._1471_since = time.monotonic()
    r._1471_short = True
    r._1456_n, r._1456_last = 0, 0.0
    s.waiting_queue = []
    s.pdflip_post_wake_settle = [r]
    return s, r, calls


def _tick(s, r):
    r._1471w_t = 0.0
    return s._pdflip_post_wake_settle_tick()


def test_review_soll4_215_through_the_settle_cohort_hold_and_a_declined_reissue(tmp_path, monkeypatch, caplog):
    """215 not released at the first tick: P's tail is published (the ack -> one re-read), the re-issue is DECLINED
    (re-stamp, record kept), the cohort holds one tick; then the settle releases it. D-NORECOMPUTE must price it on the
    record's own start: 158 -> D's own extend under the front X, no needs-p, no P leg."""
    from flliper.srt.pdflip import wake_cohort as _wc

    monkeypatch.setattr(sw, "_tail_facts", lambda rid: ("complete", False))
    holds = [1]
    monkeypatch.setattr(_wc, "asks", lambda settle, env=None: True)
    # the hold is voted the first time a member is READY (a sibling read in flight on the metal)
    monkeypatch.setattr(_wc, "hold_vote", lambda members, *a, **k: (
        holds.pop() if holds and any(flag for _r, _st, flag in members) else 0))
    s, r, calls = _settle_sched(tmp_path, monkeypatch, _rec215())
    released = 0
    with caplog.at_level(logging.INFO):
        for _ in range(6):
            released += _tick(s, r)
            if s.waiting_queue:
                break
    assert calls == [r.rid], "the metal's one declined re-issue"
    assert r._prefetch_registered_prefix_len == BASE + GOT, "the request stamp moved"
    assert not holds and any("PDFLIP-WAKE-COHORT HOLD" in m for m in caplog.messages), "the cohort held one tick"
    assert s.waiting_queue == [r], [m[:160] for m in caplog.messages if "SETTLE" in m or "NORECOMPUTE" in m]
    assert r._pdflip_store_delivered == 70841 and dn.d_would_compute(r) == 158
    assert any("D-DIRECT-EXTEND rid=" + str(r.rid) in m for m in caplog.messages)
    assert not rvp.take_requests(rvp.needs_p_dir()), "no needs-p: no P leg for a kurz request (R2/R7)"


def test_mutant_soll4_with_the_request_stamp_the_settle_sends_215_to_p(tmp_path, monkeypatch):
    """The pre-MUSS-1 reading (start from the re-stamped request): 70841 + 17244 > 70999 -> base 0 -> 53908 -> needs-p."""
    monkeypatch.setattr(sw, "_tail_facts", lambda rid: ("complete", False))
    monkeypatch.setattr(dn, "span_base", lambda req, record: (
        int(getattr(req, "_prefetch_registered_prefix_len", 0) or 0)
        if int(getattr(req, "_prefetch_registered_prefix_len", 0) or 0) + int(record.deliverable)
        <= len(req.full_untruncated_fill_ids) else 0))
    s, r, _calls = _settle_sched(tmp_path, monkeypatch, _rec215())
    for _ in range(6):
        _tick(s, r)
        if not s.pdflip_post_wake_settle:
            break
    recs = rvp.take_requests(rvp.needs_p_dir())
    assert s.waiting_queue == [] and len(recs) == 1 and recs[0]["d_extent"] == 53908
