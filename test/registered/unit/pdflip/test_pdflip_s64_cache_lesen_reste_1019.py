"""S64 (19.10.): the cache-read remainders of review S59 point 7, with the metal numbers.

Metal: NF rc15 ``boot_weg2_...dauer10190714_489909d138_1019_071429`` (D and P
logs) and rc14 ``...dauer10190107_92dd662831_1019_010742`` (D).

(S-ARM) D's store arm under-priced: rc15 pdflip-72-525 ``X-GATE-TERMS store=68608``
    at 08:31:34 while the store held 98560 (D's own ``PARK-DEMOTE role=park
    pool=anchor l3_pages=2 complete=1`` 08:30:47, P ``store_now=98560``);
    rc14 pdflip-16-211 20032 then 27200 while P read 37888. Hypothesis (i):
    the per-request cache keyed (generation, local_match, span) holds the first
    answer while the store fills under an unchanged key -- PROVEN here on the
    real ``Scheduler._pdflip_local_store_matches``. Hypothesis (ii): the probe
    caps at the L3 anchor while the END anchor is in an L2 arena slot --
    REFUTED here on the real ``HiCacheFile.batch_exists_v2`` (an arena-COMPLETE
    anchor counts before L3 is asked).
(S-CAP) rc15 pdflip-72-527 (08:30:42-52): delivered 26176 / 26048 / 25536 /
    25408 against deliverable 26560, named ``store_held_no_growth`` and #248h
    "ended short on its own capacity". The numbers are S59 class A to the
    token (TP0 span start = its device match 22208 / 26048 / 25536 / 25408,
    workers 384..1152 lower; the length MIN gave the workers TP0's page count
    from their start) -- fixed by CLAIM-END in the base. What is left is the
    name: capacity (delivered fell at the same deliverable / a room cut) is
    told apart from a store that held no more.
(S-1400) rc15 P pdflip-72-525 08:31:45: PP0 ``#915 PREFETCH REFUSED
    reason=too_short need=192 threshold=256`` at intake AND in ``P-HELD-REREAD
    ... why=wake_probe hint=98560``; told=98368; PP1/PP2 ``OVER-UNRESUMABLE
    told=98368 own=98560 resumable_at_told=0``; ``PF TOLD-FALLBACK ... -> 0
    reason=mismatch acks={1: 98560, 2: 98560}``; P recomputed 98592.
    (a) PP0's completion re-read has minimum 1; (b) unanimous follower acks
    above told that PP0's store holds -> PP0 re-reads and admits that depth
    for every rank instead of told=0.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from types import SimpleNamespace

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

PAGE = 64
_HERE = os.path.dirname(os.path.abspath(__file__))
_MGR = os.path.join(os.path.dirname(_HERE), "managers")


# =============================================================================== S-ARM


class _Probe:
    """The store as the probe sees it: a page count that can grow."""

    def __init__(self, pages):
        self.pages = pages
        self.calls = 0

    def __call__(self, tokens, last_hash, prefix_keys=None, **_kw):
        self.calls += 1
        return self.pages


def _arm_sched(probe, local_match, total):
    from flliper.srt.managers import scheduler as sch

    req = SimpleNamespace(
        rid="pdflip-72-525", full_untruncated_fill_ids=list(range(total)),
        prefix_indices=[0] * local_match, host_hit_length=0, last_host_node=None,
        extra_key=None, _compute_max_prefix_len=lambda n: n - 1,
    )
    tree = SimpleNamespace(cache_controller=SimpleNamespace(store_presence_pages=probe), page_size=PAGE,
                           is_eagle=False, hicache_storage_pass_prefix_keys=False)
    s = SimpleNamespace(tree_cache=tree, waiting_queue=[req])
    return s, sch, req


def _vote(s, sch):
    return sch.Scheduler._pdflip_local_store_matches(s, ["pdflip-72-525"])["pdflip-72-525"]


def test_arm_follows_the_store_within_one_generation_72_525():
    """RED on b5c3f1bbea: the second vote returns the cached 68608."""
    probe = _Probe(68608 // PAGE)
    s, sch, _req = _arm_sched(probe, 0, 98592)
    assert _vote(s, sch) == 68608
    probe.pages = 98560 // PAGE          # D's park demote / P's flush landed, same generation
    assert _vote(s, sch) == 98560
    assert probe.calls == 2
    assert getattr(s, "_s64_store_arm_stale", 0) == 1
    assert s._s64_store_arm_stale_tok == 98560 - 68608


def test_arm_16_211_three_votes_rc14():
    probe = _Probe(20032 // PAGE)
    s, sch, _req = _arm_sched(probe, 0, 38649)
    seen = []
    for pages in (20032, 27200, 37888):
        probe.pages = pages // PAGE
        seen.append(_vote(s, sch))
    assert seen == [20032, 27200, 37888]


def test_arm_switch_off_is_the_generation_cache(monkeypatch):
    from flliper.srt.managers import scheduler as sch

    monkeypatch.setenv(sch.S64_STORE_ARM_FRESH_ENV, "0")
    probe = _Probe(68608 // PAGE)
    s, sch, _req = _arm_sched(probe, 0, 98592)
    assert _vote(s, sch) == 68608
    probe.pages = 98560 // PAGE
    assert _vote(s, sch) == 68608 and probe.calls == 1


def test_arm_default_on():
    from flliper.srt.managers import scheduler as sch

    assert sch._s64_store_arm_fresh_on({})
    assert not sch._s64_store_arm_fresh_on({sch.S64_STORE_ARM_FRESH_ENV: "0"})


def test_arm_a_raising_probe_still_casts_an_empty_vote():
    """A vote may never break the reduce (#1317c) -- the fresh path keeps that."""
    from flliper.srt.managers import scheduler as sch

    def boom(*_a, **_k):
        raise RuntimeError("store down")

    s, sch, _req = _arm_sched(boom, 0, 98592)
    assert sch.Scheduler._pdflip_local_store_matches(s, ["pdflip-72-525"]) == {}


class _DenseBackend:
    """A KV-only backend (a KV-row worker's view): ``held`` leading pages."""

    def __init__(self, held):
        self.held = held
        self.calls = []

    def batch_exists(self, keys, extra_info=None):
        self.calls.append(len(keys))
        n = 0
        for k in keys:
            if int(k[1:]) >= self.held:
                break
            n += 1
        return n


def _worker_probe(held_pages, span_tokens, batch=1024, kv_row_worker=True):
    from flliper.srt.managers import cache_controller as cc
    from flliper.srt.managers.cache_controller import HiCacheController

    be = _DenseBackend(held_pages)
    old_cv = cc.claim_vote_min_only
    cc.claim_vote_min_only = lambda controller: kv_row_worker   # TP1/TP2 of the NF D (Form A token cut)
    ctl = SimpleNamespace(
        page_size=PAGE, storage_backend=be, _presence_pool_transfers=lambda: None,
        get_hash_str=lambda ids, last_hash, page_size: [f"h{i}" for i in range(len(list(ids)) // page_size)],
    )
    old = cc.STORAGE_BATCH_SIZE
    cc.STORAGE_BATCH_SIZE = batch
    try:
        return HiCacheController.store_presence_pages(ctl, list(range(span_tokens)), None), be
    finally:
        cc.STORAGE_BATCH_SIZE = old
        cc.claim_vote_min_only = old_cv


def test_arm_dense_group_of_equals_keeps_the_first_batch():
    """Byte-identical outside the defect locus: a dense rank that is not a
    KV-row worker beside a v2 host (27B page_size=1, any classic group) asks
    MAMBA-ANCHOR 1015's first batch, as before."""
    pages, be = _worker_probe(10 ** 6, 3000 * PAGE, batch=1024, kv_row_worker=False)
    assert pages == 1024 and be.calls == [1024]


def test_arm_root_kv_row_worker_asked_one_batch_72_525():
    """THE metal root of 68608: TP1/TP2 (KV-row workers, no mamba transfer)
    asked the dense question of the first 1024 pages only -- 3072 + 65536 =
    68608 = the group MIN store arm. RED on b5c3f1bbea (1024 pages)."""
    local_match, total = 3072, 98592
    span = (total - 1) - local_match                 # _compute_max_prefix_len = total - 1
    held = (98560 - local_match) // PAGE             # the store holds KV to 98560
    pages, be = _worker_probe(held, span)
    assert local_match + pages * PAGE == 98560
    assert be.calls == [1024, span // PAGE - 1024]   # two batches, in key order


@pytest.mark.parametrize("rid,local_match,metal", [
    ("pdflip-16-78", 960, 66496), ("pdflip-18-81", 384, 65920), ("pdflip-28-114", 768, 66304),
    ("pdflip-59-496", 0, 65536), ("pdflip-82-632", 4736, 70272), ("pdflip-104-767", 3072, 68608),
    ("pdflip-104-769", 3008, 68544),
])
def test_arm_root_reproduces_every_capped_arm_rc15(rid, local_match, metal):
    """Every rc15 W31 whose store arm sat at local_match + 1024 pages: the old
    first-batch question reproduces it, the batched walk is not capped."""
    span = 120000
    pages_old, _ = _worker_probe(10 ** 6, span, batch=1024)
    from flliper.srt.managers import cache_controller as cc

    os.environ[cc.S64_DENSE_FULL_SPAN_ENV] = "0"
    try:
        capped, _ = _worker_probe(10 ** 6, span, batch=1024)
    finally:
        os.environ.pop(cc.S64_DENSE_FULL_SPAN_ENV, None)
    assert local_match + capped * PAGE == metal
    assert pages_old == span // PAGE


def test_arm_dense_walk_stops_at_the_first_short_batch():
    pages, be = _worker_probe(1500, 400 * PAGE * 5, batch=1024)
    assert pages == 1500 and be.calls == [1024, 976]
    pages, be = _worker_probe(0, 3000 * PAGE, batch=1024)
    assert pages == 0 and be.calls == [1024]


def test_arm_dense_switch_default_on():
    from flliper.srt.managers import cache_controller as cc

    assert cc.s64_dense_full_span_on({}) and not cc.s64_dense_full_span_on({cc.S64_DENSE_FULL_SPAN_ENV: "0"})


def test_arm_full_span_answer_stays_cached():
    """Cost: an answer covering the whole span cannot grow -- no second probe."""
    probe = _Probe(10 ** 6)
    s, sch, _req = _arm_sched(probe, 0, 98592)
    assert _vote(s, sch) == 98591 and _vote(s, sch) == 98591   # clamped to the span (total - 1)
    assert probe.calls == 1


def _fake_store(n_pages, anchor_page, anchor_in_arena, anchor_on_l3):
    """The real ``batch_exists_v2`` over a fake arena / L3 index (the
    ``test_hicache_probe_chunked_1002`` double): every KV page COMPLETE in the
    arena, ONE mamba anchor at ``anchor_page``, in the arena slot and/or on L3."""
    from flliper.srt.mem_cache import hicache_storage as hs

    keys = [f"k{i:05d}" for i in range(n_pages)]
    arena = {k: 2 for k in keys}
    l3 = set()
    ak = f"{keys[anchor_page]}.mamba"
    if anchor_in_arena:
        arena[ak] = 2
    if anchor_on_l3:
        l3.add(ak)
    f = hs.HiCacheFile.__new__(hs.HiCacheFile)
    f._canonical_probe_mismatch = lambda: None
    f._arena_kv_present_prefix = lambda ks: n_pages
    f._collect_existing_component_keys = lambda chunk, pt: set()
    f._get_component_key = lambda k, name=None: k if name in (None, "__default__", hs.PoolName.KV) else f"{k}.{name}"
    f._arena_dir = lambda: "/arena"
    f._canonical_window = lambda k0: SimpleNamespace(total_bytes=1)
    f._arena_for = lambda nb: SimpleNamespace(find_states=lambda stems: [arena.get(s, 0) for s in stems],
                                              lookup=lambda stems: [arena.get(s, 0) == 2 for s in stems])
    f._suffix_for_key = lambda k: ("",)
    f._l3_index = lambda: SimpleNamespace(has=lambda stems: [s in l3 for s in stems])
    f._readable_stems = lambda stems: [s for s in stems if s in l3 or arena.get(s, 0) == 2]
    pts = [hs.PoolTransfer(name="mamba", keys=["a"], hit_policy=hs.PoolHitPolicy.TRAILING_PAGES)]
    return f, keys, pts


@pytest.mark.parametrize("chunked", ["0", "1"])
def test_arm_hypothesis_ii_refuted_l2_slot_anchor_counts(monkeypatch, chunked):
    """72-525 geometry: 1540 KV pages, the END anchor at page 1539 (98560) in the
    L2 arena slot only (P ``MAMBA-ARENA end_anchor=slot``), L3 knows nothing.
    The probe answers 1540 pages: the L2 slot is seen, (ii) is not the cause."""
    from flliper.srt.mem_cache import hicache_storage as hs

    monkeypatch.setenv(hs.PROBE_CHUNKED_ENV, chunked)
    f, keys, pts = _fake_store(1540, 1539, anchor_in_arena=True, anchor_on_l3=False)
    assert f.batch_exists_v2(keys, pts, None).kv_hit_pages == 1540
    # and an anchor nowhere caps it (the double is honest)
    f2, keys2, pts2 = _fake_store(1540, 1539, anchor_in_arena=False, anchor_on_l3=False)
    assert f2.batch_exists_v2(keys2, pts2, None).kv_hit_pages == 0


# =============================================================================== S-CAP


def _s59():
    sys.path.insert(0, _HERE)
    sys.path.insert(0, _MGR)
    return importlib.import_module("test_pdflip_s59_d_admission_store_1019")


#: rc15 D TP0, pdflip-72-527: (host span start = TP0 device match, worker span
#: start = TP1/TP2 device match, metal completed_synced) -- deliverable 26560.
READS_72_527 = [
    (22208, 21824, 26176),   # 08:30:42 matched=22208 loaded=3968 / workers matched=21824
    (26048, 25536, 26048),   # 08:30:43 matched=26048 loaded=0 / workers 25536
    (25536, 24512, 25536),   # 08:30:46
    (25408, 24256, 25408),   # 08:30:46
]
END_72_527 = 26560


@pytest.mark.parametrize("hb,wb,metal", READS_72_527)
def test_cap_72_527_is_claim_end_not_capacity(hb, wb, metal):
    """The four 'capacity' reads are the length-MIN claim cut, to the token: the
    old reduce reproduces each metal completion, CLAIM-END (S59, in the base)
    lands every rank at the store's END 26560 with its anchor."""
    s59 = _s59()
    bases = {0: hb, 1: wb, 2: wb}
    claims = {0: END_72_527 - hb, 1: END_72_527 - wb, 2: END_72_527 - wb}
    old_end = min(bases[r] + min(claims.values()) for r in range(3))
    assert old_end == metal, "the old length reduce reproduces the metal completion"
    ctls, ops, errors, _g = s59._claim_round("pdflip-72-527", bases, claims, hb)
    assert errors == {} and all(not x.stops for x in ctls.values())
    own = {r: int(ops[r].probed_hit_tokens) for r in range(3)}
    assert s59._group_end(bases, own) == END_72_527


def test_cap_names_capacity_apart_from_store_shortness():
    from flliper.srt.pdflip import short_read_cause as c

    # 72-527: 26176 -> 26048 at deliverable 26560: the reader lost, the store did not
    assert c.classify(26048, 26560, 26176, 26560, None, held=True, settled=False) == c.CAPACITY_REGRESS
    assert c.classify(25408, 26560, 25536, 26560, None, True, False) == c.CAPACITY_REGRESS
    assert c.classify(26048, 26560, 26048, 26560, "arena_full", True, False) == c.CAPACITY_CUT
    assert c.classify(26048, 26560, 26048, 26560, "late_complete", True, False) == c.STORE_HELD
    assert c.classify(26048, 26560, 26048, 26560, None, False, False) == c.NO_WRITER
    assert c.classify(26048, 26560, 26176, 26560, None, True, True) == c.SETTLED
    # the store grew (new deliverable): never a capacity regress
    assert c.classify(26048, 30000, 26176, 26560, None, True, False) == c.STORE_HELD


def test_cap_fallback_line_carries_the_split_name(monkeypatch, caplog):
    """The real ``_pdflip_store_short_fallback`` prints the classified reason;
    the decision (``expired``) is unchanged."""
    from flliper.srt.managers import scheduler as sch
    from flliper.srt.pdflip import short_read_cause as c

    monkeypatch.setattr(sch, "_pdflip_store_short_cycle", lambda s, r: 2)
    monkeypatch.setattr(sch, "_pdflip_settled_this_wake", lambda s, r: False)
    monkeypatch.setattr(sch, "_pdflip_store_held_short", lambda r: True)
    monkeypatch.setattr(sch, "_pdflip_store_short_tail_x", lambda s: 12288)
    monkeypatch.setattr(sch, "_pdflip_store_short_remainder", lambda r: 544)
    monkeypatch.setattr(sch, "_pdflip_windowed_path", lambda s: False)
    s = SimpleNamespace(_clear_prefetch_deferral_fields=lambda r: None)
    req = SimpleNamespace(rid="pdflip-72-527")
    c.note_record(req, 26176, 26560, None)
    req._pdflip_store_delivered, req._pdflip_store_deliverable = 26176, 26560
    c.note_record(req, 26048, 26560, None)
    req._pdflip_store_delivered = 26048
    import logging

    with caplog.at_level(logging.WARNING, logger=sch.logger.name):
        out = sch._pdflip_store_short_fallback(s, req, sch._DEFER_REASON_STORE_SHORT, 26591, "drain")
    assert out == "expired"
    line = [r.getMessage() for r in caplog.records if "PREFETCH-DEFER-FALLBACK" in r.getMessage()][-1]
    assert "reason=capacity_regress" in line and "prev_delivered=26176" in line
    assert s._s64_short_causes == {c.CAPACITY_REGRESS: 1}


# =============================================================================== S-1400 (a)


class _P0Tree:
    def __init__(self, have):
        self.have = have

    def completed_prefetch_tokens(self, rid):
        return self.have


def _held_sched(monkeypatch, store_now=98560):
    from flliper.srt.managers import pdflip_store_told as st

    seen = {}

    def _prefetch(req):
        # the #915 gate of 72-525: need 192 (98368 -> 98560) against threshold 256
        mn = st.completion_read_min_tokens(req)
        seen["min"] = mn
        need, threshold = 192, 256
        return "issued" if need >= (mn if mn is not None else threshold) else "declined:too_short"

    s = SimpleNamespace(page_size=PAGE, pdflip_dormant=False, _prefetch_kvcache=_prefetch)
    monkeypatch.setattr(st, "_wake_store_depth", lambda sched, req, have: store_now)
    req = SimpleNamespace(rid="pdflip-72-525")
    setattr(req, st.WAKE_PROBE_ATTR, 1)
    return st, s, req, seen


def test_1400a_wake_reread_reads_the_192_token_remainder(monkeypatch):
    """RED on b5c3f1bbea: ``verdict=declined:too_short`` -- the told stays 98368."""
    st, s, req, seen = _held_sched(monkeypatch)
    from flliper.srt.pdflip import short_read as sr

    sr.forget("pdflip-72-525")
    assert st._p_held_short_reread(s, _P0Tree(0), req, "pdflip-72-525") is True
    assert seen["min"] == 1
    assert getattr(req, st.COMPLETION_READ_ATTR, None) is None   # the mark lived for that one call
    assert st.completion_read_min_tokens(req) is None
    sr.forget("pdflip-72-525")


def test_1400a_switch_off_is_the_threshold(monkeypatch):
    st, s, req, seen = _held_sched(monkeypatch)
    from flliper.srt.pdflip import short_read as sr

    monkeypatch.setenv(st.COMPLETION_READ_ENV, "0")
    sr.forget("pdflip-72-525")
    assert st._p_held_short_reread(s, _P0Tree(0), req, "pdflip-72-525") is False
    assert seen["min"] is None
    sr.forget("pdflip-72-525")


def test_1400a_mark_cleared_when_the_read_raises():
    from flliper.srt.managers import pdflip_store_told as st

    req = SimpleNamespace(rid="x")

    def boom(r):
        assert st.completion_read_min_tokens(r) == 1
        raise RuntimeError("io")

    with pytest.raises(RuntimeError):
        st.issue_completion_read(boom, req)
    assert st.completion_read_min_tokens(req) is None


def test_1400a_prefetch_kvcache_takes_the_completion_minimum():
    """Wiring: ``_prefetch_kvcache`` folds the completion minimum into the
    read's ``min_tokens`` beside park / told / hand-back."""
    import inspect

    from flliper.srt.managers import scheduler as sch

    src = inspect.getsource(sch.Scheduler._prefetch_kvcache)
    i = src.index("pdflip_store_told.completion_read_min_tokens(req)")
    j = src.index('_tail_kw = {"min_tokens": _tail_min}')
    assert i < j


# =============================================================================== S-1400 (b)


def _raise_ring(monkeypatch, pp1_own, pp2_own, store=98560, pp0_resumes=True, completion_ok=True):
    """The PF ring (paced told + ack stream + follower early read) with the
    72-525 depths: PP0 reads 98368, the followers ``pp1_own`` / ``pp2_own``;
    a follower's tree resumes only at its own read's end (OVER-UNRESUMABLE)."""
    sys.path.insert(0, _MGR)
    import _told_ring_pf as R  # noqa: E402

    from flliper.srt.managers import pdflip_store_told as m
    from flliper.srt.managers import pdflip_told_fallback as fb
    from flliper.srt.managers import pdflip_told_raise as rz

    rid, told = "aaaa-told", 98368
    monkeypatch.delenv("FLLIPER_PDFLIP_TOLD_FIDELITY", raising=False)
    for k in list(os.environ):
        if k.startswith("FLLIPER_PDFLIP_TOLD") or k in ("FLLIPER_PDFLIP_P_TWIN_DEFER", m.ENV_FOLLOWER_EARLY_READ):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(fb.ENV_FALLBACK, "1")
    monkeypatch.delenv("FLLIPER_PDFLIP_DUAL_SHARE", raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_DUAL_LAYOUT", raising=False)
    monkeypatch.setenv(m.ENV_FOLLOWER_EARLY_READ, "1")
    monkeypatch.setattr(m, "_absolute_armed", lambda: True)
    ring = R.Ring(m, monkeypatch, {rid: told}, {r: {rid: (0.8, 0.6, 0.6)[r]} for r in range(3)})
    ring.stages[1].prompts = {rid: pp1_own}
    ring.stages[2].prompts = {rid: pp2_own}
    p0 = ring.stages[0]
    base = p0._prefetch_kvcache
    p0.reads = []

    def _p0_read(req, rematch=True, limit_tokens=None):
        mn = m.completion_read_min_tokens(req)
        p0.reads.append(mn)
        if mn == 1 and completion_ok:
            p0.prompts = {rid: store}       # the completion read takes what the store holds
        return base(req, rematch, limit_tokens)

    p0._prefetch_kvcache = _p0_read
    owns = {1: pp1_own, 2: pp2_own}

    def resumable(scheduler, req, depth):
        r = scheduler.ps.pp_rank
        if r == 0:
            return None
        own = owns[r]
        return min(int(depth), own) if int(depth) >= own else 0

    def pp0_admissible(scheduler, req, depth):
        have = int(scheduler.tree_cache.completed.get(rid, 0) or 0)
        return int(depth) if (pp0_resumes and have >= int(depth)) else min(have, int(depth)) if pp0_resumes else 0

    monkeypatch.setattr(m._tf, "rank_resumable", resumable)
    monkeypatch.setattr(m._tf, "pp0_admissible", pp0_admissible)
    monkeypatch.setattr(rz, "store_anchored_depth", lambda sched, req: store)
    return ring, rid, told, m, fb, rz


def test_1400b_unanimous_acks_above_told_raise_on_every_rank_72_525(monkeypatch):
    """RED on b5c3f1bbea: ``PF TOLD-FALLBACK told=98368 -> 0 reason=mismatch
    acks={1: 98560, 2: 98560}``; P recomputed 98592."""
    ring, rid, told, m, fb, rz = _raise_ring(monkeypatch, 98560, 98560)
    ring.arrive(rid)
    ring.run(160)
    plans = ring.plans(rid)
    assert all(len(p) == 1 for p in plans) and plans[0] == plans[1] == plans[2], plans
    assert plans[0][0][2] == 98560, plans                   # cap 98560 on every rank, P computes 32
    assert getattr(ring.stages[0], "_pf_fallback_n", 0) == 0
    assert ring.stages[0]._s64_raise_admit_n == 1
    assert ring.stages[0].reads.count(1) == 1                # one completion read
    assert ring.stages[0].tree_cache.released == []          # PP0's own read was not released
    admit = [o for o in ring.wire_objs() if type(o).__name__ == "PdFlipStoreAdmit"][-1]
    assert admit.told == 98560 and getattr(admit, rz.WIRE_RAISE) == 1
    assert not getattr(admit, fb.WIRE_FALLBACK, 0)
    for s in ring.stages:
        assert s.tree_cache.op_refs == 0 and rid not in s.tree_cache.ongoing


@pytest.mark.parametrize("pp1,pp2,store,resumes,ok,why", [
    (98560, 98496, 98560, True, True, "acks_differ"),        # rang-uneinig: no raise
    (98560, 98560, 98368, True, True, "store_short"),        # store does not hold the acked depth
    (98560, 98560, 98560, False, True, "pp0_not_resumable"),
    (98560, 98560, 98560, True, False, "read_short"),        # PP0's completion read lands short
])
def test_1400b_no_raise_without_proof_is_told_zero_as_before(monkeypatch, pp1, pp2, store, resumes, ok, why):
    ring, rid, told, m, fb, rz = _raise_ring(monkeypatch, pp1, pp2, store=store, pp0_resumes=resumes,
                                             completion_ok=ok)
    ring.arrive(rid)
    ring.run(400)
    plans = ring.plans(rid)
    assert all(len(p) == 1 for p in plans) and plans[0] == plans[1] == plans[2], (why, plans)
    assert plans[0][0][2] == 0, (why, plans)               # the unchanged PF answer, on every rank
    assert getattr(ring.stages[0], "_s64_raise_admit_n", 0) == 0
    for s in ring.stages:
        assert s.tree_cache.op_refs == 0 and rid not in s.tree_cache.ongoing


def test_1400b_partial_acks_wait_only_within_the_frist(monkeypatch):
    """PP1 acked 98560 above told, PP2's read never ends (its ack never comes):
    the raise waits only inside the unchanged PF Frist, then told=0 for every
    rank -- no hang."""
    ring, rid, told, m, fb, rz = _raise_ring(monkeypatch, 98560, 98560)
    ring.stages[2].read_s = {rid: 10 ** 6}
    ring.arrive(rid)
    ring.run(1500)
    plans = ring.plans(rid)
    assert all(len(p) == 1 for p in plans) and plans[0] == plans[1] == plans[2], plans
    assert plans[0][0][2] == 0
    assert getattr(ring.stages[0], "_s64_raise_admit_n", 0) == 0


def test_1400b_switch_off_is_told_zero(monkeypatch):
    ring, rid, told, m, fb, rz = _raise_ring(monkeypatch, 98560, 98560)
    monkeypatch.setenv(rz.ENV_RAISE, "0")
    ring.arrive(rid)
    ring.run(160)
    plans = ring.plans(rid)
    assert plans[0] == plans[1] == plans[2] and plans[0][0][2] == 0
    assert ring.stages[0]._pf_fallback_n == 1


def test_1400b_raise_wait_is_bounded(monkeypatch):
    """A raise read that never terminates: the PF rule decides after the bound."""
    from flliper.srt.managers import pdflip_told_raise as rz

    acks = {1: 98560, 2: 98560}
    tree = SimpleNamespace(check_prefetch_progress=lambda rid: False)
    s = SimpleNamespace(ps=SimpleNamespace(pp_size=3), tree_cache=tree, _prefetch_kvcache=lambda req: "issued")
    monkeypatch.setattr(rz, "store_anchored_depth", lambda sched, req: 98560)
    req = SimpleNamespace(rid="r")
    assert rz.pp0_raise_step(s, "r", req, 98368, acks, True, now=0.0) == (rz.WAIT, 98560)
    assert rz.pp0_raise_step(s, "r", req, 98368, acks, True, now=rz.wait_s() - 0.1) == (rz.WAIT, 98560)
    assert rz.pp0_raise_step(s, "r", req, 98368, acks, True, now=rz.wait_s() + 0.1) is None
    assert rz.pp0_raise_step(s, "r", req, 98368, acks, True, now=99.0) is None   # one attempt per rid


def test_1400b_candidate_rule():
    from flliper.srt.managers import pdflip_told_raise as rz

    assert rz.raise_candidate({1: 98560, 2: 98560}, 98368, 3) == 98560
    assert rz.raise_candidate({1: 98560}, 98368, 3) is None             # an ack missing
    assert rz.raise_candidate({1: 98560, 2: 98496}, 98368, 3) is None   # acks differ
    assert rz.raise_candidate({1: 98368, 2: 98368}, 98368, 3) is None   # equal to told: the normal admit
    assert rz.raise_candidate({1: 3072, 2: 3072}, 98368, 3) is None     # below told: H64 floor's business
    assert rz.raise_on({}) and not rz.raise_on({rz.ENV_RAISE: "0"})
    # the metal order: PP1's ack landed a pass before PP2's -- a raise is still possible
    assert rz._partial_candidate({1: 98560}, 98368, 3) == 98560
    assert rz._partial_candidate({1: 98368}, 98368, 3) is None
    assert rz._partial_candidate({1: 98560, 2: 98560}, 98368, 3) is None   # complete: raise_candidate's


def test_1400b_only_pp0_decides():
    """R11/#1004: the raise lives in PP0's paced publish only; the follower
    side reads the wire marker for its log line and nothing else."""
    import inspect

    from flliper.srt.managers import pdflip_store_told as m

    assert "pp0_raise_step" in inspect.getsource(m._pp0_publish_paced)
    for fn in (m._follower_admit, m._follower_absorb_impl, m.admission, m.follower_early_settle):
        assert "pp0_raise_step" not in inspect.getsource(fn)
