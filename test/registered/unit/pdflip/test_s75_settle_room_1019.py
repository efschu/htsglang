"""S75 (19.10.): two waits on rc16 (bf9ca46f45, boot 1019_121629, D TP0).

(1) SETTLE waited for a writer that does not exist. pdflip-190-722 was D-direct
(14:20:39 ``#1324c ... no hand-off chain, no hand-off record``); D's park retract
wrote a #1442 record of its OWN chain (``cache_finished_req`` ->
``_pdflip_handoff_write`` runs on D too); at the wake 14:22:27 the read was short
(40000 of 44032) and the settle read D's record as P's: ``SETTLE-WRITER
writer=p-handoff action=poll``, 2-s re-reads, the 6th D seat held 21 s until
``W50-REROUTE`` 14:22:48. Records now name their writer group; D's own record is
no P writer; a P record bounds what P can still write (P-span).

(2) pdflip-137-471 (144965 tokens) waited 193 s behind the older pdflip-132-465:
the pressure-park early resume read the FREE ids only (#1045 FLOOR PUBLISHED
floor=113536 .. 104128 < need 144965) while the arrival rule, the displacement
and the adder count the tree's evictable tokens too. It now reads the group MIN of
available + payable evictable, and ``PDFLIP D-PARK-RESUME-HELD`` names the gate.
"""
import json
import logging
import os
import time
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers import scheduler as sched_mod  # noqa: E402
from flliper.srt.pdflip import d_park_runtime as dpr  # noqa: E402
from flliper.srt.pdflip import d_seats as ds  # noqa: E402
from flliper.srt.pdflip import handoff as ho  # noqa: E402
from flliper.srt.pdflip import handoff_keys as hk  # noqa: E402
from flliper.srt.pdflip import settle_writer as sw  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="stage-a-pdflip-unit")

S = sched_mod.Scheduler
X = 12288
PAGE = 64


# ---------------------------------------------------------------- (1) settle writer


class _Refetch:
    def __init__(self):
        self.reissued = []

    def __call__(self, req, now, allow_reissue=False):
        req._1471_short = True
        if now - float(getattr(req, "_1456_last", 0.0) or 0.0) < 2.0:
            return "wait"
        if not allow_reissue:
            return "due"
        req._1456_last = now
        self.reissued.append(req.rid)
        return "reissued"


def _holder():
    h = types.SimpleNamespace()
    h.waiting_queue = []
    h.pdflip_dormant_hold = []
    h.tree_cache = types.SimpleNamespace(cache_controller=types.SimpleNamespace(pdflip_hold_rids=set()),
                                         prefetch_loaded_tokens_by_reqid={})
    h.PDFLIP_POST_WAKE_SETTLE_S = S.PDFLIP_POST_WAKE_SETTLE_S
    h.ps = types.SimpleNamespace(tp_size=1)
    h.page_size = PAGE
    h.server_args = types.SimpleNamespace(tp_prefill_max_tokens=X)
    for n in ("_pdflip_post_wake_settle_tick", "_pdflip_group_min_flags"):
        setattr(h, n, types.MethodType(getattr(S, n), h))
    h._pdflip_refetch_one = _Refetch()
    return h


def _req(rid, n, delivered):
    r = types.SimpleNamespace(rid=rid, full_untruncated_fill_ids=list(range(n)),
                              _pdflip_store_delivered=delivered, _prefetch_span_tokens=n)
    r._1471_since = time.monotonic()
    r._1456_last = time.monotonic()
    r._1471_short = True
    return r


@pytest.fixture
def arena(tmp_path, monkeypatch):
    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.setenv("FLLIPER_PDFLIP_STORE_SHORT_TAIL", "0")
    monkeypatch.delenv("FLLIPER_PDFLIP_GROUP", raising=False)
    monkeypatch.delenv(sw.LABEL_ENV, raising=False)
    (tmp_path / "handoff").mkdir()
    return tmp_path


def _lines(caplog, tag):
    return [r.getMessage() for r in caplog.records if tag in r.getMessage()]


def _keys(n):
    return [f"k{i}" for i in range(n)]


def test_the_record_names_its_writer_group(arena, monkeypatch):
    assert ho.write("pdflip-1-1", [1, 2], ["a"])                   # outside a pdflip group: no field
    assert json.loads((arena / "handoff" / "pdflip-1-1.json").read_text()) == {"input_ids": [1, 2],
                                                                               "page_keys": ["a"]}
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    assert ho.write("pdflip-1-1", [1, 2], ["a"])
    assert ho.record_writer(ho.read("pdflip-1-1")) == "D"
    assert ho.write("pdflip-1-1", [1, 2], ["a"], writer="p")
    assert ho.record_writer(ho.read("pdflip-1-1")) == "P"
    assert ho.record_writer({"input_ids": []}) == "" and ho.record_writer(None) == ""


def test_the_metal_form_ds_own_record_is_no_writer_decided_at_the_first_tick(arena, caplog):
    # 190-722: D-direct, D's retract wrote its own 689-page chain; the wake read 40000 of 44032
    rid = "pdflip-190-722"
    assert ho.write(rid, list(range(44161)), _keys(689), writer="D")
    r = _req(rid, 44161, 40000)
    assert hk.resolve_chain(r, ho.read) == _keys(689)               # the #248 wake read takes the keys
    assert getattr(r, hk.SEEN_ATTR) is True and getattr(r, hk.P_SEEN_ATTR) is False
    h = _holder()
    h.pdflip_post_wake_settle = [r]
    with caplog.at_level(logging.INFO, logger=sched_mod.logger.name):
        assert h._pdflip_post_wake_settle_tick() == 1
    assert h.waiting_queue == [r] and h._pdflip_refetch_one.reissued == []
    w = _lines(caplog, "SETTLE-WRITER")
    assert len(w) == 1 and "writer=none" in w[0] and "action=decide not_p=d-own-record" in w[0]
    assert "remainder=4161" in _lines(caplog, "SETTLE-NO-WRITER")[0]


def test_base_form_switch_off_d_record_still_polls(arena, monkeypatch):
    monkeypatch.setenv(sw.LABEL_ENV, "0")
    rid = "pdflip-174-648"
    assert ho.write(rid, list(range(32847)), _keys(513), writer="D")
    r = _req(rid, 32847, 27712)
    hk.resolve_chain(r, ho.read)
    assert sw.observe(r, page_size=PAGE) == sw.P_HANDOFF            # the rc16 view: p-handoff, poll


def test_a_p_record_keeps_the_poll_while_ps_span_is_not_delivered(arena):
    rid = "pdflip-5-5"
    assert ho.write(rid, list(range(41021)), _keys(640), writer="P")   # P's prefill: 640 pages
    r = _req(rid, 41021, 40512)                                       # 633 pages delivered
    hk.resolve_chain(r, ho.read)
    assert getattr(r, hk.P_SEEN_ATTR) is True and getattr(r, hk.P_PAGES_ATTR) == 640
    assert sw.observe(r, page_size=PAGE) == sw.P_HANDOFF
    assert getattr(r, sw.WHY_ATTR) is None


def test_ps_span_delivered_means_p_has_nothing_left(arena, caplog):
    rid = "pdflip-6-6"
    assert ho.write(rid, list(range(40000)), _keys(625), writer="P")    # P wrote [0, 40000)
    r = _req(rid, 44161, 40000)                                       # D decoded on, parked, read 40000
    hk.resolve_chain(r, ho.read)
    assert ho.write(rid, list(range(44161)), _keys(689), writer="D")    # D's retract overwrote the file
    assert sw.observe(r, page_size=PAGE) == sw.NONE
    assert getattr(r, sw.WHY_ATTR) == "p-span-delivered"
    assert getattr(r, hk.P_SEEN_ATTR) is True, "P's record was seen once: P stays a candidate"
    assert sw.observe(r, page_size=None) == sw.P_HANDOFF, "no page size: the bound is not applied"
    r._pdflip_store_delivered = 39936
    assert sw.observe(r, page_size=PAGE) == sw.P_HANDOFF               # one P page still missing


def test_the_token_span_guards_a_page_count_that_runs_short(arena):
    rid = "pdflip-6-7"
    assert ho.write(rid, list(range(40100)), _keys(625), writer="P")    # 626 whole pages of prompt
    r = _req(rid, 44161, 40000)
    hk.resolve_chain(r, ho.read)
    assert getattr(r, hk.P_TOKENS_ATTR) == 40100
    assert sw.observe(r, page_size=PAGE) == sw.P_HANDOFF
    r._pdflip_store_delivered = 40064
    assert sw.observe(r, page_size=PAGE) == sw.NONE


def test_a_legacy_record_without_the_field_reads_as_p_as_before(arena):
    rid = "pdflip-7-7"
    p = arena / "handoff" / f"{rid}.json"
    p.write_text(json.dumps({"input_ids": [0] * 10, "page_keys": _keys(3)}))
    r = _req(rid, 41021, 40512)
    hk.resolve_chain(r, ho.read)
    assert getattr(r, hk.P_SEEN_ATTR) is True and getattr(r, hk.P_PAGES_ATTR, None) is None
    assert sw.observe(r, page_size=PAGE) == sw.P_HANDOFF


def test_a_chain_without_any_record_read_is_p_as_before(arena):
    r = _req("pdflip-8-8", 41021, 40512)
    setattr(r, hk.CHAIN_ATTR, ["k0", "k1"])
    assert sw.observe(r, page_size=PAGE) == sw.P_HANDOFF


def test_d_own_record_with_ps_tail_parts_under_write_still_waits(arena):
    rid = "pdflip-9-9"
    assert ho.write(rid, [0] * 10, _keys(1), writer="D")
    (arena / "handoff" / f"{rid}.tail.pp2-1.pt.77.1.0.tmp").write_bytes(b"x")
    r = _req(rid, 41021, 40512)
    assert sw.observe(r, page_size=PAGE) == sw.P_WRITING


def test_reset_for_wake_clears_the_why(arena):
    r = _req("pdflip-10-10", 10, 5)
    setattr(r, sw.WHY_ATTR, "d-own-record")
    sw.reset_for_wake(r)
    assert getattr(r, sw.WHY_ATTR) is None


# ---------------------------------------------------------------- (2) resume room gate

ON = {"FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE": "1"}
NEED = 144965           # pdflip-137-471 on the metal (CAP-LIFT incoming=144965)
FREE = 104128           # #1045 FLOOR PUBLISHED floor at 13:30:56
EVICT = 290000          # the finished 138-472/138-473 spans the tree held (192000 + 99584 loaded)


def _sreq(rid, seq, *, site=None, n_in=100):
    r = types.SimpleNamespace(rid=rid, kv_arrival_seq=seq, origin_input_ids=[0] * n_in,
                              output_ids=[], is_fast_lane=False, spill_class=None)
    if site is not None:
        ds.mark_parked(r, site, epoch=7, now=100.0)
    return r


def _gate(book, young, old, avail):
    return ds.admission_gate([young], running=[old], avail_tokens=avail, resume_book=book)


def test_metal_numbers_free_only_hold_it_for_good_free_plus_evictable_resumes():
    book = ds.ResumeBook.from_env(ON)
    old = _sreq("pdflip-132-465", 465, n_in=116317)
    young = _sreq("pdflip-137-471", 471, site=ds.SITE_PRESSURE, n_in=NEED)
    base = [_gate(book, young, old, FREE) for _ in range(3 * book.steps)]
    assert {g.skip(young) for g in base} == {"pdflip_d_park_older_live"}
    assert base[0].held == (("pdflip-137-471", "older_live", "pdflip-132-465", FREE, NEED),)
    book = ds.ResumeBook.from_env(ON)
    got = [_gate(book, young, old, lambda: FREE + EVICT).skip(young) for _ in range(book.steps)]
    assert got[:-1] == ["pdflip_d_park_older_live"] * (book.steps - 1) and got[-1] is None


def test_the_reading_is_lazy_and_read_once_per_pass():
    calls = []

    def avail():
        calls.append(1)
        return 10 ** 9

    book = ds.ResumeBook.from_env(ON)
    a = _sreq("a", 1)
    ds.admission_gate([a], running=[], avail_tokens=avail, resume_book=book)
    assert calls == [], "no pressure-parked request asks: no read (no collective)"
    old = _sreq("old", 0)
    y1 = _sreq("y1", 2, site=ds.SITE_PRESSURE)
    y2 = _sreq("y2", 3, site=ds.SITE_PRESSURE)
    ds.admission_gate([y1, y2], running=[old], avail_tokens=avail, resume_book=book)
    assert calls == [1], "two candidates, one read"
    ds.admission_gate([_sreq("f", 4, site=ds.SITE_FLIP)], running=[old], avail_tokens=avail, resume_book=book)
    assert calls == [1], "a flip park never asks the early resume"


def test_held_for_older_is_named():
    book = ds.ResumeBook.from_env(ON)
    older = _sreq("older", 1)
    y = _sreq("y", 2, site=ds.SITE_PRESSURE)
    setattr(y, ds.DISPLACED_FOR_ATTR, "older")
    g = ds.admission_gate([older, y], running=[], avail_tokens=10 ** 9, resume_book=book)
    assert g.held == (("y", "held_for_older", "older", None, None),)


class _Alloc:
    def __init__(self, n):
        self.n = n

    def available_size(self):
        return self.n


class _Tree:
    def __init__(self, n):
        self.n = n

    def evictable_size(self):
        return self.n


def test_the_reading_is_the_group_min_of_free_plus_evictable(monkeypatch):
    monkeypatch.delenv(dpr.RESUME_FIT_EVICTABLE_ENV, raising=False)
    seen = []
    sched = types.SimpleNamespace(token_to_kv_pool_allocator=_Alloc(FREE), tree_cache=_Tree(EVICT),
                                  uniform_min_avail=lambda: FREE)
    sched._pdflip_group_min_ints = lambda vals: (seen.append(list(vals)), [vals[0] - 1000, vals[1]])[1]
    read = dpr._resume_fit_reading(sched)
    # KVOFFSET-1024 (class E): the reading is this rank's LIFETIME budget as the largest price the gate admits
    # (``ceil(rem_total_tokens) - 1``: the gate refuses at price >= budget); nothing runs here, so no reserve
    assert read() == FREE + EVICT - 1 - 1000 and seen == [[FREE + EVICT - 1, 1]]
    monkeypatch.setenv(dpr.RESUME_FIT_EVICTABLE_ENV, "0")
    assert dpr._resume_fit_reading(sched)() == FREE, "switch off: the free ids, as before"


def test_the_held_line_names_gate_and_numbers_on_powers_of_two(caplog):
    sched = types.SimpleNamespace()
    gate = types.SimpleNamespace(held=(("pdflip-137-471", "older_live", "pdflip-132-465", FREE, NEED),))
    with caplog.at_level(logging.INFO, logger=dpr.logger.name):
        for _ in range(5):
            dpr._note_resume_held(sched, gate)
    lines = _lines(caplog, dpr.RESUME_HELD_MARK)
    assert [l.split("passes=")[1].split()[0] for l in lines] == ["1", "2", "4"]
    assert "gate=older_live older=pdflip-132-465 avail=104128 need=144965" in lines[0]
    dpr._note_resume_held(sched, types.SimpleNamespace(held=()))
    assert sched._pdflip_resume_held_seen == {}


# ---------------------------------------------------------------- review S75 (Auflagen 1.1-1.3, 2.1)


class _RaisingAlloc:
    def available_size(self):
        raise RuntimeError("allocator gone")


def test_2_1_a_rank_without_a_reading_still_enters_the_min_exactly_once(monkeypatch, caplog):
    """3 ranks, rank 1's allocator raises. Each rank must enter the group MIN once (its peers do);
    the fake collective answers every rank with the element-wise MIN of all three payloads."""
    monkeypatch.delenv(dpr.RESUME_FIT_EVICTABLE_ENV, raising=False)
    payloads = {0: [FREE + EVICT - 1, 1], 1: [0, 0], 2: [FREE + EVICT - 1, 1]}   # KVOFFSET-1024: largest price
    group_min = [min(col) for col in zip(*payloads.values())]
    entered, sent = {0: 0, 1: 0, 2: 0}, {}

    def make(i):
        def gm(vals):
            entered[i] += 1
            sent[i] = list(vals)
            return list(group_min)
        return gm

    scheds = [types.SimpleNamespace(token_to_kv_pool_allocator=_RaisingAlloc() if i == 1 else _Alloc(FREE),
                                    tree_cache=_Tree(EVICT), uniform_min_avail=lambda: FREE,
                                    _pdflip_group_min_ints=make(i)) for i in range(3)]
    with caplog.at_level(logging.INFO, logger=dpr.logger.name):
        results = [dpr._resume_fit_reading(s)() for s in scheds]
    assert entered == {0: 1, 1: 1, 2: 1}, "every rank enters the group MIN exactly once"
    assert sent == payloads, "the rank without a reading sends the sentinel (ok=0)"
    assert results == [None, None, None], "the group refuses the early resume alike, by name"
    assert any("UNREAD" in m for m in _lines(caplog, dpr.RESUME_HELD_MARK))


class _RaisingLedger:
    def outstanding_tokens(self):
        raise RuntimeError("ledger gone")


def test_2_1b_a_rank_whose_budget_terms_raise_still_enters_the_min_exactly_once(monkeypatch):
    """KVOFFSET-1024 review A2 (S75 2.1 for the lifetime reading): the resume reading now reads more terms than the pool
    (dcp deficit, ledger, decode reserve). A rank that raises on ANY of them sends the sentinel into the one MIN --
    never leaves before the collective (a peer alone in the gloo MIN hangs: W17)."""
    monkeypatch.delenv(dpr.RESUME_FIT_EVICTABLE_ENV, raising=False)

    def _boom():
        raise RuntimeError("deficit gone")

    bad_req = types.SimpleNamespace(rid="bad", output_ids=None, finished=lambda: False,
                                    sampling_params=types.SimpleNamespace(max_new_tokens=100))
    entered, sent = {}, {}

    def make(i):
        def gm(vals):
            entered[i] = entered.get(i, 0) + 1
            sent[i] = list(vals)
            return [0, 0]
        return gm

    kinds = {
        0: dict(uniform_budget_deficit=_boom),
        1: dict(chunked_commitment_ledger=_RaisingLedger()),
        2: dict(running_batch=types.SimpleNamespace(reqs=[bad_req])),
    }
    results = []
    for i, extra in kinds.items():
        s = types.SimpleNamespace(token_to_kv_pool_allocator=_Alloc(FREE), tree_cache=_Tree(EVICT),
                                  uniform_min_avail=lambda: FREE, _pdflip_group_min_ints=make(i), **extra)
        results.append(dpr._resume_fit_reading(s, getattr(s, "running_batch", None))())
    assert entered == {0: 1, 1: 1, 2: 1}, "every rank enters the group MIN exactly once"
    assert all(v[1] == 0 for v in sent.values()), "a rank whose terms raise sends the sentinel (ok=0)"
    assert results == [None, None, None]


def _p_then_d(rid, delivered):
    assert ho.write(rid, list(range(6400)), _keys(100), writer="P")    # P prefilled, record read at admission
    r = _req(rid, 7000, delivered)
    assert hk.resolve_chain(r, ho.read)
    assert ho.write(rid, list(range(7000)), _keys(109), writer="D")    # D's park overwrote the file
    return r


def test_1_2_p_chain_then_d_overwrite_stays_poll_and_names_nothing(arena):
    r = _p_then_d("pdflip-9-9", 3200)
    assert sw.observe(r, page_size=PAGE) == sw.P_HANDOFF
    assert getattr(r, hk.P_SEEN_ATTR) is True
    assert getattr(r, sw.WHY_ATTR) is None, "1.1: no not_p= while the verdict is p-handoff"


def test_1_2_mutant_p_seen_from_the_current_file_is_red(arena, monkeypatch):
    """Mutant: P_SEEN read from the CURRENT file instead of sticky -> the danger direction opens."""
    r = _p_then_d("pdflip-9-12", 3200)

    def non_sticky(req, rec, chain):
        setattr(req, hk.P_SEEN_ATTR, ho.record_writer(rec) != "D")
    monkeypatch.setattr(hk, "note_record_writer", non_sticky)
    assert sw.observe(r, page_size=PAGE) == sw.NONE, "the mutant decides a P-prefilled rid -- the test above catches it"


def test_1_2_the_settle_line_of_a_d_file_over_a_p_chain_says_poll_without_not_p(arena, caplog):
    r = _p_then_d("pdflip-9-13", 3200)
    r._1471_since = time.monotonic()
    r._1456_last = time.monotonic()
    r._1471_short = True
    h = _holder()
    h.pdflip_post_wake_settle = [r]
    with caplog.at_level(logging.INFO, logger=sched_mod.logger.name):
        assert h._pdflip_post_wake_settle_tick() == 0
    w = _lines(caplog, "SETTLE-WRITER")
    assert len(w) == 1 and "writer=p-handoff" in w[0] and "action=poll --" in w[0] and "not_p" not in w[0]


def test_1_3_a_p_leg_without_a_fresh_p_record_keeps_the_poll(arena):
    from flliper.srt.pdflip import resume_via_p as rvp

    rid = "pdflip-9-11"
    assert ho.write(rid, list(range(7000)), _keys(109), writer="D")
    r = _req(rid, 7000, 3200)
    hk.resolve_chain(r, ho.read)
    assert sw.observe(r, page_size=PAGE) == sw.NONE                    # before any P leg: decided
    setattr(r, rvp.ATTEMPTS_ATTR, 1)                                   # RESUME-VIA-P ran, P wrote no record
    sw.reset_for_wake(r)
    assert sw.observe(r, page_size=PAGE) == sw.P_HANDOFF
    assert getattr(r, sw.WHY_ATTR) is None
