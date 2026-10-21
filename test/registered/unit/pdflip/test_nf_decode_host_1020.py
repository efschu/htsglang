# SPDX-License-Identifier: Apache-2.0
"""nf-decode-host-1020: the D decode pauses of an admission (D-DECODE-GAP cause=other).

Metal (rc28 D ``..._297198745d_1020_174205``, TP0, all three ranks alike)::

    18:43:48 Decode rank batch #round 44274 t 1792521828.677 bs 1          <- the opener
    18:43:48 PDFLIP D-IDS rid=pdflip-22-334 n=101450                       <- an agent turn arrives
    18:43:48 D-TRANSIENT-LEND return rows_on 22->19 (short extend pending ...) ms=86
    18:43:48 LIVE-SPANS ... order=return-short unmap_ms=31.0 / fence-wait ms=31.1
    18:43:48 D-TRANSIENT-LEND stage-keep S4->S6 rows_on 19->18 (S106)
    18:43:49 D-MEM-SCHED from=S4 to=S6 used=101466 incoming=101450 -- grow to hold 211598
    18:43:49 #969 EXTENT reqs=[('pdflip-22-334', 101248, 101450, 101248, 202)]
    18:43:49 D-LEAD-MS rid=pdflip-22-334 recv_to_extend_ms=560.8
    18:43:49 D-DECODE-GAP gap_ms=586 cause=other rows=1 run_ms=23 host_ms=563 notes=lend_return:2 return_ms=118

The 202-token extend shares 101248 tokens of tree pages with the running ``pdflip-22-333``; the census
counted them twice and grew the KV stage two steps (a live move in the decode pause, an end-event
shrink later). 106 of the boot's 116 grows that the next ``#969 EXTENT`` names were held by the stage
they left once the shared prefix counted once.
"""
from __future__ import annotations

import logging
import types

import pytest

from flliper.srt.pdflip import d_decode_gap as G
from flliper.srt.pdflip import d_mem_share as S

try:
    from flliper.test.ci.ci_register import register_cpu_ci

    register_cpu_ci(est_time=5, suite="base-a-test-cpu")
except Exception:  # noqa: BLE001 -- a desk without the CI registry
    pass

GRID = [32768, 65536, 98304, 131072, 163840, 196608, 229376, 262144, 393216, 524288]
CLIP = 4096
PAGE = 64

# the shared conversation prefix of pdflip-22-333 / pdflip-22-334 (distinct token ids, then each turn's own)
_PREFIX = list(range(1000, 1000 + 101248))


def _req(rid, ids, protected=0, n_out=0, max_new=0):
    return types.SimpleNamespace(
        rid=rid, origin_input_ids=list(ids), output_ids=[7] * n_out, finished=lambda: False,
        cache_protected_len=protected, sampling_params=types.SimpleNamespace(max_new_tokens=max_new))


def _r333(protected=101248):
    # 101295 prompt + 171 decoded = 101466 (the D-MEM-SCHED line's used)
    return _req("pdflip-22-333", _PREFIX + [5] * 47, protected=protected, n_out=171)


def _r334(protected=101248, priced=202):
    # 101450 tokens, 202 new (#969 EXTENT); the front's header: X-GATE front_x=7140 front_priced=202
    r = _req("pdflip-22-334", _PREFIX + [9] * 202, protected=protected)
    if priced is not None:
        r.weg2_front_x = 7140
        r.weg2_front_priced = priced
    return r


class _Group:
    """The D group's CPU collective: element-wise MIN over what every rank contributes to the SAME call.
    Records each call's vector length per rank (a different length = a split collective)."""

    def __init__(self, n):
        self.n = n
        self.pending = {}
        self.calls = {r: [] for r in range(n)}

    def gmin_for(self, rank):
        def gmin(vals):
            vals = [int(v) for v in vals]
            self.calls[rank].append(len(vals))
            self.pending[rank] = vals
            return vals  # replaced below by _settle once every rank contributed
        return gmin


@pytest.fixture
def env(monkeypatch):
    from flliper.srt.pdflip import d_seat_vram as dsv

    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.setenv("FLLIPER_OPT_PDFLIP_D_SEAT_VRAM", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_D_KV_STAGE_TOKENS", ",".join(str(t) for t in GRID))
    monkeypatch.setenv("FLLIPER_PDFLIP_D_KV_STAGE_BY_DEMAND", "1")
    monkeypatch.setenv("FLLIPER_CLIP_MAX_NEW_TOKENS_ESTIMATION", str(CLIP))
    monkeypatch.delenv(S.ENV, raising=False)
    monkeypatch.setattr(dsv, "_engage_kv_cap", lambda alloc, t, p: t)
    monkeypatch.setattr(dsv, "max_live_page", lambda alloc: 0)
    # S77-B's grow-room vote is its own group call (not under test here): every rank has room
    monkeypatch.setattr(dsv, "_grow_room_refused", lambda *a, **k: None)

    def make(running, queue, gmin=None, stage=4):
        sched = types.SimpleNamespace(
            server_args=types.SimpleNamespace(max_running_requests=6, chunked_prefill_size=4096,
                                              speculative_num_draft_tokens=4, page_size=PAGE),
            running_batch=types.SimpleNamespace(reqs=list(running), batch_is_full=False),
            waiting_queue=list(queue), chunked_req=None, last_batch=None, page_size=PAGE,
            token_to_kv_pool_allocator=object(),
            new_token_ratio_tracker=types.SimpleNamespace(current=0.0))
        if gmin is not None:
            sched._pdflip_group_min_ints = gmin
        setattr(sched, dsv.CTL_ATTR, False)
        setattr(sched, dsv.PHASE_ATTR, dsv.PhaseState(epoch="e22", n=6, cap=6, done=True,
                                                      stage=stage, stage_tokens=GRID[stage]))
        return sched

    return dsv, make


def _stage(dsv, s):
    return getattr(s, dsv.PHASE_ATTR).stage


# ---------------------------------------------------------------- the LCP


def test_lcp_shapes():
    a = list(range(10000))
    assert S.lcp(a, a) == 10000
    assert S.lcp(a, a[:5000]) == 5000
    b = list(a)
    b[6000] = -1
    assert S.lcp(a, b) == 6000
    b = list(a)
    b[0] = -1
    assert S.lcp(a, b) == 0
    assert S.lcp([], a) == 0
    assert S.lcp(tuple(a), iter(a[:4097])) == 4097


# ---------------------------------------------------------------- the census


def test_the_metal_case_counts_the_shared_prefix_once(env):
    dsv, make = env
    s = make([_r333()], [_r334()])
    used, incoming, sh_u, sh_in = S.census(s, [_r333()], [_r334()], PAGE)
    assert (used, incoming) == (101466, 202)
    assert (sh_u, sh_in) == (0, 101248)


def test_the_metal_case_holds_s4(env):
    """18:43:49: base grew S4 -> S6 ("grow to hold 211598"); counted once the need is 101466 + 202 + air +
    reserve, which S4 (163840) holds -- no live move in the admission's decode pause."""
    dsv, make = env
    s = make([_r333()], [_r334()])
    step = dsv.runtime_tick(s)
    assert _stage(dsv, s) == 4
    assert step is not None and not step.changed


def test_switch_off_is_the_old_count(env, monkeypatch):
    dsv, make = env
    monkeypatch.setenv(S.ENV, "0")
    s = make([_r333()], [_r334()])
    assert S.census(s, [_r333()], [_r334()], PAGE) == (101466, 101450, 0, 0)
    dsv.runtime_tick(s)
    assert _stage(dsv, s) == 6  # the metal line: from=S4 to=S6


def test_no_tree_prefix_no_credit(env):
    """A request without a committed tree prefix (no arrival match) gets no credit: the old count."""
    dsv, make = env
    s = make([_r333()], [_r334(protected=0)])
    assert S.census(s, s.running_batch.reqs, s.waiting_queue, PAGE)[:2] == (101466, 101450)


def test_credit_is_capped_by_both_tree_prefixes_and_page_floored(env):
    dsv, make = env
    s = make([], [])
    run = _r333(protected=90944 + 10)  # TP-local tree prefix of the running one: 90954 -> page floor 90944
    used, incoming, _, sh_in = S.census(s, [run], [_r334()], PAGE)
    assert sh_in == 90944 and incoming == 101450 - 90944
    s = make([], [])
    used, incoming, _, sh_in = S.census(s, [_r333()], [_r334(protected=63)], PAGE)
    assert sh_in == 0 and incoming == 101450  # less than one page shared: nothing


def test_without_the_front_header_nothing_is_a_candidate(env):
    """The candidate list is built from replicated inputs only (token ids + the front's carried price): a queued
    request without the header is no candidate -- no credit, and the reserve call keeps its single value."""
    dsv, make = env
    calls = []
    s = make([_r333()], [_r334(priced=None)], gmin=lambda v: calls.append(list(v)) or list(v))
    pl = S.prepare(s, s.running_batch.reqs, s.waiting_queue, PAGE)
    assert pl.cands == [] and pl.local == []
    dsv.runtime_tick(s)
    assert _stage(dsv, s) == 6
    assert calls and all(len(c) == 1 for c in calls), calls


def test_the_front_cached_count_caps_the_credit(env):
    dsv, make = env
    s = make([], [])
    # the front priced 50000 of 101450 uncached: at most 51450 cached anywhere -> floor to the page
    used, incoming, _, sh_in = S.census(s, [_r333()], [_r334(priced=50000)], PAGE)
    assert sh_in == (51450 // PAGE) * PAGE


def test_different_prompts_share_nothing(env):
    dsv, make = env
    s = make([], [])
    other = _req("pdflip-9-1", list(range(5, 5 + 101450)), protected=101248)
    other.weg2_front_x, other.weg2_front_priced = 7140, 202
    assert S.census(s, [_r333()], [other], PAGE)[:2] == (101466, 101450)


# ---------------------------------------------------------------- rank agreement (raenge-nie-uneins)


def _tick_group(dsv, make, protected_by_rank, refused_rank=None):
    """Three ranks tick the same replicated lists; the queued request's arrival match differs per rank. The
    collective is simulated in lock-step: every rank's vector of the ONE reserve+credit call is collected
    first, then each rank gets the element-wise MIN."""
    group = _Group(3)
    scheds = {}
    for r, p in protected_by_rank.items():
        s = make([_r333()], [_r334(protected=p)])
        if refused_rank == r:
            S.note_refused(s, s.waiting_queue[0])
        scheds[r] = s
    # pass 1: what every rank puts into the call (the reserve agreement's vector with the credits)
    vecs = {}
    for r, s in scheds.items():
        running, admissible = dsv._demand_lists(s)
        pl = S.prepare(s, running, admissible, PAGE)
        vecs[r] = [-dsv.admission_reserve(s, running, admissible)] + list(pl.local)
    lens = {len(v) for v in vecs.values()}
    assert len(lens) == 1, vecs
    answer = [min(v[k] for v in vecs.values()) for k in range(lens.pop())]
    # pass 2: the real tick; the reserve+credit call answers with the group MIN
    for r, s in scheds.items():
        def gmin(vals, _r=r):
            vals = [int(v) for v in vals]
            group.calls[_r].append(len(vals))
            return list(answer) if len(vals) == len(answer) and vals[0] < 0 else vals
        s._pdflip_group_min_ints = gmin
        dsv.runtime_tick(s)
    return {r: _stage(dsv, s) for r, s in scheds.items()}, group, scheds


def test_rank_local_arrival_matches_take_one_stage(env):
    """TP0 matched 101248, TP1 only 90944 (a host-backed anchor, RU in tp_match_floor): the credit is the
    group MIN, every rank steps on the same need and takes the same stage, the call has one length."""
    dsv, make = env
    stages, group, scheds = _tick_group(dsv, make, {0: 101248, 1: 90944, 2: 101248})
    assert len(set(stages.values())) == 1, stages
    lens = {r: tuple(c) for r, c in group.calls.items()}
    assert len(set(lens.values())) == 1, lens
    for s in scheds.values():
        assert S.book(s).agreed[S._key("pdflip-22-333", "pdflip-22-334")] == 90944


def test_one_rank_without_a_tree_prefix_voids_the_credit(env):
    dsv, make = env
    stages, _g, scheds = _tick_group(dsv, make, {0: 101248, 1: 0, 2: 101248})
    assert stages == {0: 6, 1: 6, 2: 6}  # the old grow on every rank


def test_a_no_token_refusal_on_one_rank_voids_the_credit(env):
    """The danger direction: a credit the admission could not fund must not hold the stage below it --
    the refusing rank contributes 0, the MIN is 0, the stage grows for the full length on every rank."""
    dsv, make = env
    stages, _g, _s = _tick_group(dsv, make, {0: 101248, 1: 101248, 2: 101248}, refused_rank=1)
    assert stages == {0: 6, 1: 6, 2: 6}


# ---------------------------------------------------------------- the book after the admission


def test_running_pair_reads_the_agreed_book_and_pays_no_collective(env):
    dsv, make = env
    calls = []

    def gmin(vals):
        calls.append(list(vals))
        return list(vals)

    a, b = _r333(), _r334()
    s = make([a], [b], gmin=gmin)
    dsv.runtime_tick(s)
    assert _stage(dsv, s) == 4
    # the queued tick paid ONE call: the reserve agreement carrying the credit
    assert calls == [[-64, 101248]]
    # admitted: both run, the queue is empty -> no collective, the shared prefix still counts once
    s.running_batch.reqs = [a, b]
    s.waiting_queue = []
    n_calls = len(calls)
    used, incoming, sh_u, _ = S.census(s, [a, b], [], PAGE)
    assert len(calls) == n_calls
    assert (used, incoming, sh_u) == (101466 + 202, 0, 101248)
    dsv.runtime_tick(s)
    assert _stage(dsv, s) == 4


def test_running_pair_without_an_agreed_value_gets_no_credit(env):
    """Two running requests that never met in the queue (a wake resume): their tree prefixes are rank-local
    -- nothing local is credited, only the book."""
    dsv, make = env
    s = make([], [], gmin=lambda v: list(v))
    used, _, sh_u, _ = S.census(s, [_r333(), _r334()], [], PAGE)
    assert sh_u == 0 and used == 101466 + 101450


def test_the_earlier_one_ends_the_later_one_counts_whole(env):
    dsv, make = env
    a, b = _r333(), _r334()
    s = make([a], [b], gmin=lambda v: list(v))
    S.census(s, [a], [b], PAGE)
    assert S.census(s, [a, b], [], PAGE)[0] == 101466 + 202
    # pdflip-22-333 finished: its pair leaves the book, pdflip-22-334 holds the pages alone
    assert S.census(s, [b], [], PAGE)[0] == 101450
    assert S.book(s).agreed == {}
    # and does not come back if the rid were seen again without a new agreement
    assert S.census(s, [a, b], [], PAGE)[0] == 101466 + 101450


def test_no_queue_no_collective(env):
    dsv, make = env
    calls = []
    s = make([_r333()], [], gmin=lambda v: calls.append(list(v)) or list(v))
    S.census(s, s.running_batch.reqs, [], PAGE)
    dsv.runtime_tick(s)
    assert calls == []


def test_cap_lift_keeps_the_plain_sums(env, monkeypatch):
    """The cap-lift verdict (a pending shrink's cap) weighs the plain sums -- its conservative side."""
    dsv, make = env
    seen = {}

    def spy(sched, ms, fr, alloc, tokens, used, incoming, admissible, rids, page):
        seen["v"] = (used, incoming)
        return False, 0, 0

    monkeypatch.setattr(dsv, "_cap_lift_verdict", spy)
    s = make([_r333()], [_r334()])
    dsv.runtime_tick(s)
    assert seen["v"] == (101466, 101450)


def test_the_stage_line_names_the_credit(env, caplog):
    dsv, make = env
    s = make([_r333()], [_r334()], stage=3)  # S3 (131072) holds 101466 + 202 + air + reserve
    with caplog.at_level(logging.INFO):
        dsv.runtime_tick(s)
    lines = [r.getMessage() for r in caplog.records if "D-MEM-SHARE" in r.getMessage()]
    assert lines and "shared_incoming=101248" in lines[0] and "raw_incoming=101450" in lines[0]


# ---------------------------------------------------------------- D-DECODE-GAP spans


@pytest.fixture
def gap_env(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    for e in (G.ENV, G.ENV_MS, G.ENV_CENSUS_S):
        monkeypatch.delenv(e, raising=False)


def test_spans_name_the_host_time_of_the_gap():
    tr = G.Tracker(400, 0, 0.0)
    tr.on_batch(0.0, 1, 0, 1)
    tr.on_run_end(0.023, 1)
    tr.span("result", 5.0)
    tr.span("recv", 24.0)
    tr.span("lend_start", 180.0)
    tr.span("seat_stage", 220.0)
    tr.span("admit", 120.0)
    tr.span("match", 30.0)  # nested in admit
    tr.span("lend_end", 1.0)
    rec = tr.on_batch(0.586, 1, 203, 1)
    assert rec["cause"] == "other" and round(rec["host_ms"]) == 563
    txt = G.format_spans(rec)
    assert txt.startswith("recv:24,lend_start:180,seat_stage:220,admit:120,match:30,lend_end:1,result:5")
    assert txt.endswith("rest:13")  # 563 - (24+180+220+120+1+5); match is nested
    # the next gap starts empty
    rec2 = tr.on_batch(1.2, 1, 0, 1)
    assert G.format_spans(rec2) == "-"


def test_spans_inside_an_extend_window_are_the_extend(gap_env):
    tr = G.Tracker(400, 0, 0.0)
    tr.on_batch(0.0, 2, 0, 1)
    tr.span("admit", 50.0)
    tr.on_batch(0.1, 0, 188, 1)  # a pure extend opens its window
    tr.span("admit", 999.0)      # inside the extend's wall: d_prefill_ms already holds it
    rec = tr.on_batch(1.4, 2, 0, 1)
    assert rec["spans"] == {"admit": 50.0}


def test_span_before_any_decode_round_is_dropped():
    tr = G.Tracker(400, 0, 0.0)
    tr.span("admit", 10.0)
    assert tr.spans == {}


def test_span_helper_is_rank_local_and_never_raises(gap_env, caplog):
    s = types.SimpleNamespace(tp_rank=0, _pdflip_wake_seq=1)
    G.span(s, "recv", 0.0)  # before a round: dropped
    G.span(s, "recv", "not-a-number")  # never raises
    off = types.SimpleNamespace(tp_rank=1, _pdflip_wake_seq=1)
    G.span(off, "recv", 0.0)
    assert G.tracker(off) is None


def test_gap_line_carries_spans(gap_env, caplog):
    s = types.SimpleNamespace(tp_rank=0, _pdflip_wake_seq=3)

    class _M:
        def __init__(self, k):
            self.k = k

        def is_decode(self):
            return self.k == "d"

        def is_extend(self):
            return self.k == "e"

    dec = types.SimpleNamespace(forward_mode=_M("d"), batch_size=lambda: 1, reqs=[types.SimpleNamespace(rid="a")])
    mixed = types.SimpleNamespace(forward_mode=_M("e"), extend_num_tokens=203,
                                  decoding_reqs=[types.SimpleNamespace(rid="a")])
    with caplog.at_level(logging.INFO, logger=G.__name__):
        G.on_batch(s, dec, now=100.0)
        G.on_run_end(s, dec, now=100.02)
        G.tracker(s).span("lend_start", 200.0)
        G.on_batch(s, mixed, now=100.6)
    line = [r.getMessage() for r in caplog.records if r.getMessage().startswith(G.MARK + " gap_ms")][0]
    assert "spans=lend_start:200,rest:380" in line


def test_scheduler_wires_the_spans():
    """The span sites sit where the scheduler already passes (source check of the wiring)."""
    import inspect

    from flliper.srt.managers import scheduler as sch

    src = inspect.getsource(sch)
    for name in ("recv", "result", "lend_start", "seat_stage", "admit", "match", "lend_end"):
        assert '_pdflip_d_decode_gap.span(self, "%s"' % name in src, name
    assert "_pdflip_d_mem_share.note_refused(self, req)" in src


def test_reserve_and_credits_ride_one_call_element_wise(env):
    """The reserve stays the group MAX (y6h) and each credit the group MIN, in the one vector."""
    dsv, make = env
    peer = [-300, 4000]
    s = make([], [_r334()], gmin=lambda v: [min(a, b) for a, b in zip(v, peer)])
    assert dsv._agree_reserve(s, s.waiting_queue, 100, credits=[5000]) == (300, [4000])
    # without credits: the old int, the old call
    s2 = make([], [_r334()], gmin=lambda v: [min(v[0], -300)])
    assert dsv._agree_reserve(s2, s2.waiting_queue, 100) == 300


def test_a_short_answer_from_the_group_credits_nothing(env):
    """A group answer without the credit values (a peer on the old vector) must not leave the local credit
    standing: nothing is credited, the old grow."""
    dsv, make = env
    s = make([_r333()], [_r334()], gmin=lambda v: [v[0]])
    dsv.runtime_tick(s)
    assert _stage(dsv, s) == 6
