"""KVOFFSET-1024 (SOLL v2 R5, R7, R11; follow-up of front-handover-kvfits-1024 + review S2/S3/S5).

D publishes the terms of its lifetime gate in the SAME KV reading the front decides on (``pdflip_kv.lifetime_budget``
and its parts, pdflip/d_lifetime_reading.py); the front takes it by min() for ARRIVAL-SEAT and the D line alike; the
burst debit covers every KV test; D's pressure-park resume prices like the gate. Metal specimens: 27B int4ar
``_1023_223241`` / ``_1024_003732`` (front + D logs under /spinning/docker-acceptance/27b/evidence).

Danger directions, one test group each:
  * Uebergabe trotz vollem KV  -- a (offset, 25-238), b (deliverable evictable), c (publisher == adder), d (group
                                  ratio), e (ARRIVAL-SEAT burst, 34-281/282/283), f (credit only for a predecessor D
                                  runs), g (D retracted its decodes: 30-296), h (unread does not release a KV waiter,
                                  32-276), i (resume test = gate, 34-283), j (resume reading = MIN of own budgets)
  * Verhungern                 -- k (old D unchanged), l (KV ladder keeps its room), m/n (admitted / ledger rids not
                                  charged twice), o (the priced need is charged, not the bare prompt), p (S3: D runs
                                  none of the seated rids), q (a victim gives its reserve back), r (resume when it
                                  fits), h2 (a first tick without a number still goes: R2 / EZ-S1 A10)
"""
from __future__ import annotations

import asyncio
import math
import os
import time
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers import schedule_policy as sp  # noqa: E402
from flliper.srt.pdflip import arrival_seat_rule as asr  # noqa: E402
try:  # the red run (before the fix) has no module: every test that needs it fails by name, not the whole file
    from flliper.srt.pdflip import d_lifetime_reading as dlr  # noqa: E402
except ImportError:  # pragma: no cover - red run only
    dlr = None
from flliper.srt.pdflip import d_park_runtime as dpr  # noqa: E402
from flliper.srt.pdflip import d_seats as ds  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip.front import Front, Seat  # noqa: E402

CLIP = sp.CLIP_MAX_NEW_TOKENS


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_PBOUND_FLIP_NOW", "0")
    for k in ("FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_AGE_PLAN", "FLLIPER_PDFLIP_SEAT_ORDER_AFTER_FLIP",
              "FLLIPER_PDFLIP_SHORT_GATE_BY_AGE", "FLLIPER_PDFLIP_KV_FRESH_MS", "FLLIPER_PDFLIP_EXACT_CLASS",
              dpr.RESUME_FIT_EVICTABLE_ENV):
        monkeypatch.delenv(k, raising=False)


# ----------------------------------------------------------------------------------------------- D-side doubles
def _req(rid, n_in, out=0, max_new=64000, mamba=None):
    return SimpleNamespace(rid=rid, origin_input_ids=list(range(n_in)), output_ids=list(range(out)),
                           sampling_params=SimpleNamespace(max_new_tokens=max_new), mamba_pool_idx=mamba,
                           finished=lambda: False)


class _Alloc:
    def __init__(self, avail):
        self.avail = avail

    def available_size(self):
        return self.avail


class _Tree:
    """Plain tree: the gate reads ``evictable_size``."""

    uniform_avail_floor = None

    def __init__(self, evict):
        self.evict = evict

    def supports_mamba(self):
        return False

    def evictable_size(self):
        return self.evict


class _HybridTree(_Tree):
    """Hybrid SSM (27B): the gate reads the peel's DELIVERABLE evictable, not the reported count."""

    def __init__(self, reported, deliverable):
        super().__init__(reported)
        self.deliverable = deliverable

    def supports_mamba(self):
        return True

    def full_evictable_size(self):
        return self.evict

    def deliverable_evictable_size(self):
        return self.deliverable


class _Ledger:
    def __init__(self, d):
        self.d = dict(d)

    def outstanding_tokens(self):
        return sum(self.d.values())

    def request_ids(self):
        return list(self.d)


def _sched(avail, tree, running=(), ratio=0.5, group=None, ledger=None, page=1, mixed=False, chunked=None):
    s = SimpleNamespace(token_to_kv_pool_allocator=_Alloc(avail), tree_cache=tree,
                        running_batch=SimpleNamespace(reqs=list(running)),
                        new_token_ratio_tracker=SimpleNamespace(current=ratio), page_size=page,
                        is_mixed_chunk=mixed, chunked_req=chunked, last_batch=None,
                        chunked_commitment_ledger=ledger or _Ledger({}), uniform_budget_deficit=lambda: 0,
                        kv_session_offload=None)
    if group is not None:
        s._kvoffset_group_ratio = group
    return s


def _reading(sched, base=None):
    """``pdflip_kv`` as get_internal_state builds it: the legacy fields plus the lifetime terms."""
    got = dict(base or {})
    got.setdefault("available", sched.token_to_kv_pool_allocator.available_size())
    got.setdefault("evictable", sched.tree_cache.evictable_size())
    got.setdefault("decode_clip", CLIP)
    got.setdefault("running_rids", [r.rid for r in sched.running_batch.reqs])
    got.update(dlr.lifetime_reading(sched))
    return got


# ======================================================================== A: Uebergabe trotz vollem KV
def test_a_d_decode_reserve_refuses_what_available_plus_evictable_would_pass_25_238():
    """pdflip-25-238 (27B _1023_223241, D 23:04:08): price=51758 budget=43063.8 = available 58466 - offset 15402.
    The front's need 47661 + 4096 = 51757 <= 58466 passed (87 refusals, 23.4 s). With D's lifetime room it waits."""
    old = {"available": 58466, "evictable": 0}
    assert asr.kv_fits(51757, old)[0] is True                  # the base: passes the front
    rd = dict(old, lifetime_budget=int(math.ceil(43063.8)) - 1, decode_reserve=15402, ledger_outstanding=0,
              price_extra=1)
    need = 51757 + asr.price_extra(rd)
    assert need == 51758                                       # = D's price (extend + clip + page)
    fits, why = asr.kv_fits(need, rd)
    assert fits is False and "D lifetime room 43063" in why
    assert asr.kv_free(rd) == 43063


def test_b_hybrid_reported_evictable_is_not_room_only_the_deliverable_is():
    tree = _HybridTree(reported=80000, deliverable=10000)
    s = _sched(20000, tree)
    rd = _reading(s)
    assert rd["deliverable_evictable"] == 10000
    assert rd["lifetime_budget"] == 20000 + 10000 - 1          # nothing runs: no reserve; strict gate (price < budget)
    assert asr.kv_fits(50000, rd)[0] is False                  # the reported count said 100000 free
    assert asr.kv_fits(29999, rd)[0] is True


def _adder(sched, num_mixed=0):
    from flliper.srt.mem_cache.base_prefix_cache import DecLockRefResult, IncLockRefResult
    from flliper.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    tc = MagicMock()
    tc.supports_mamba.return_value = False
    tc.evictable_size.return_value = sched.tree_cache.evictable_size()
    tc.full_evictable_size.return_value = sched.tree_cache.evictable_size()
    tc.swa_evictable_size.return_value = 0
    tc.disable = False
    tc.uniform_avail_floor = None
    tc.inc_lock_ref.return_value = IncLockRefResult()
    tc.dec_lock_ref.return_value = DecLockRefResult()
    alloc = MagicMock()
    alloc.available_size.return_value = sched.token_to_kv_pool_allocator.available_size()
    alloc.full_available_size.return_value = alloc.available_size.return_value
    alloc.swa_available_size.return_value = 0
    rb = MagicMock()
    rb.reqs = list(sched.running_batch.reqs)
    return sp.PrefillAdder(
        page_size=1, tree_cache=tc, token_to_kv_pool_allocator=alloc, running_batch=rb,
        new_token_ratio=sched.new_token_ratio_tracker.current, rem_input_tokens=10**9, rem_chunk_tokens=None,
        num_mixed_decode_tokens=num_mixed, priority_scheduling_preemption_threshold=0,
        commitment_ledger=sched.chunked_commitment_ledger,
    )


def test_c_published_budget_is_the_adders_own_rem_total_tokens():
    """One formula: what D publishes is what the adder of the next pass computes (mixed rows, running reserve,
    ledger) -- the largest price it admits is ceil(rem_total_tokens) - 1."""
    running = [_req("r1", 98000, out=10, max_new=64000), _req("r2", 30000, out=4000, max_new=6000)]
    s = _sched(150000, _Tree(7000), running, ratio=0.4321, mixed=True, ledger=_Ledger({"ch": 1234}))
    rd = _reading(s)
    adder = _adder(s, num_mixed=len(running))
    assert rd["lifetime_budget"] == int(math.ceil(adder.rem_total_tokens)) - 1
    assert rd["decode_reserve"] == int(math.ceil(adder.rem_total_token_offset))
    assert rd["ledger_outstanding"] == 1234 and rd["ledger_rids"] == ["ch"]
    assert rd["decode_reserve_by_rid"] == {"r1": int(math.ceil(CLIP * 0.4321)), "r2": int(math.ceil(2000 * 0.4321))}


def test_c2_adder_price_is_the_shared_lifetime_price():
    r = _req("x", 47661, max_new=64000)
    assert sp.lifetime_price(47661, r, 1) == 51758             # 25-238's price
    assert sp.decode_clip_of(r) == CLIP
    assert sp.running_request_reserve(r, 0.5) == CLIP * 0.5


def test_d_reserve_is_priced_with_the_group_max_ratio_not_tp0s():
    """NF y6h 10011531: TP0's ratio priced 4964, TP1/TP2 6592. Each rank's gate subtracts its own reserve and the group
    admits only when all admit: the binding reserve is the LARGEST -- the group ratio of the per-iteration reduce."""
    running = [_req("r1", 50000, max_new=64000)]
    s = _sched(100000, _Tree(0), running, ratio=0.10, group=0.30)
    rd = _reading(s)
    assert rd["decode_reserve_scope"] == "group" and rd["decode_reserve_ratio"] == pytest.approx(0.30)
    assert rd["decode_reserve"] == int(math.ceil(CLIP * 0.30))
    s2 = _sched(100000, _Tree(0), running, ratio=0.10)          # no reduce published (offload path): TP0's own
    assert _reading(s2)["decode_reserve_scope"] == "tp0"
    assert _reading(s2)["decode_reserve"] == int(math.ceil(CLIP * 0.10))
    s3 = _sched(100000, _Tree(0), running, ratio=0.70, group=0.30)   # TP0 reset since the reduce: never below its own
    assert _reading(s3)["decode_reserve"] == int(math.ceil(CLIP * 0.70))


def test_d2_the_ratio_rides_the_per_iteration_reduce_negated_and_is_read_back_by_its_index():
    import inspect

    from flliper.srt.managers import scheduler as sch

    src = inspect.getsource(sch.Scheduler._update_uniform_pool_budget)
    assert "_ratio_at = len(vals)" in src
    assert "vals = vals + [-_dlr.ratio_ppm(self.new_token_ratio_tracker.current)]" in src
    assert "self._kvoffset_group_ratio = _dlr.ratio_from_ppm(-int(t[_ratio_at]))" in src
    # before the tail-indexed ballot, after the S42 hold pair
    assert src.index("_s42_hold_at = len(vals)") < src.index("_ratio_at = len(vals)") < src.index("_ballot_rids = [")
    assert dlr.ratio_from_ppm(-min(-dlr.ratio_ppm(0.1), -dlr.ratio_ppm(0.3))) == pytest.approx(0.3)
    src2 = inspect.getsource(sch.Scheduler.get_internal_state)
    assert "lifetime_reading(self)" in src2


# ---------------------------------------------------------------- front harness
def _front(box):
    f = Front("http://p", "http://d", "D", "kvo", "", 0, 0, {}, 45.0, tp_prefill_max_tokens=12288)
    f.awake, f.state, f.admit_d = "D", "serving", True

    async def _kv():
        return box.get("kv")

    async def _no_pool():
        return None

    f._arrival_seat_kv_reading = _kv
    f._d_reading_if_armed = _no_pool
    return f


async def _seat(f, rid, tokens, ledger=True):
    await f._d_seat.acquire()
    if ledger:
        f.groups["D"].outstanding[rid] = time.time()
    return Seat(f, rid, "short", tokens=tokens)


def test_e_three_arrival_seat_grants_are_not_priced_against_one_reading_34_281_282_283():
    """27B 23:15:59,478-,562: 34-281 (50248), 34-282 (4641, credited against 281) and 34-283 (90936, credited 18104)
    granted within 84 ms against free=114674 -- 145825 in total; 34-283 then 568 lifetime refusals on D."""

    async def body():
        kv = {"available": 114674, "evictable": 0, "decode_clip": 4096, "running_rids": ["run-a"],
              "displaceable_rids": ["run-a"]}
        f = _front({"kv": kv})
        await _seat(f, "run-a", 30000)                           # D runs it: no debit
        fits, why, _ = await Front._arrival_seat_kv(f, 64119 - 17967, 64000, "pdflip-34-281", 0)
        assert fits is True
        f._asr_st()["granted"]["pdflip-34-281"] = time.time()
        Front._asr_note_kv_priced(f, "pdflip-34-281")
        await _seat(f, "pdflip-34-281", 64119)                   # seated, D has not run it yet
        fits, why, _ = await Front._arrival_seat_kv(f, 104944, 64000, "pdflip-34-283", 0)
        assert fits is False, why                                 # 114674 - 50248 = 64426 < 109040
        assert "D-line debit 50248 of 1 seat(s)" in why

    asyncio.run(body())


def test_f_no_shared_credit_against_a_predecessor_d_does_not_run_yet():
    async def body():
        kv = {"available": 114674, "evictable": 0, "decode_clip": 4096, "running_rids": [], "displaceable_rids": []}
        f = _front({"kv": kv})
        f.groups["D"].outstanding["pdflip-34-281"] = time.time()   # granted + seated: in the front's ledger
        f.__dict__["_sess_prev"] = {"pdflip-34-282": ("pdflip-34-281", 63648)}
        assert Front._asr_shared_credit(f, "pdflip-34-282", 64193, 545, {"run-a"}) == 0
        assert Front._asr_shared_credit(f, "pdflip-34-282", 64193, 545, {"pdflip-34-281"}) == 63648
        assert Front._asr_shared_credit(f, "pdflip-34-282", 64193, 545, None) == 63648   # an old D: as before
        fits, why, _ = await Front._arrival_seat_kv(f, 64193, 64000, "pdflip-34-282", 545)
        assert "shared 0" in why

    asyncio.run(body())


def test_g_d_retracted_both_decodes_the_whole_pool_reads_free_30_296():
    """27B _1024_003732 01:18:25: D retracted 30-294 and 26-255 for one pass (spans retained, evictable), the front read
    free=278852 = the whole pool and granted 30-296 (need 104440); D re-admitted both and refused it by gate=lifetime.
    Both still hold front seats that D does not run in the reading: their rows come off the room."""

    async def body():
        kv = {"available": 586, "evictable": 278266, "decode_clip": 4096, "running_rids": [], "displaceable_rids": []}
        f = _front({"kv": kv})
        await _seat(f, "pdflip-30-294", 76247)
        await _seat(f, "pdflip-26-255", 136960)
        fits, why, _ = await Front._arrival_seat_kv(f, 100344, 64000, "pdflip-30-296", 824)
        assert fits is False, why

    asyncio.run(body())


def test_h_an_absent_reading_does_not_release_a_waiter_that_waits_for_kv_32_276():
    async def body():
        box = {"kv": None}
        f = _front(box)
        st = f._asr_st()
        st["waiters"]["pdflip-32-276"] = time.time() - 158.0
        st.setdefault("wait_named", {})["pdflip-32-276"] = "kv"
        fits, why, _ = await Front._arrival_seat_kv(f, 59069, 64000, "pdflip-32-276", 0)
        assert fits is False and "kv=unread held" in why
        assert f.counters["arrival_seat_kv_unread_held"] == 1

    asyncio.run(body())


def test_h2_a_first_tick_without_a_number_still_goes_r2_ez_s1_a10():
    async def body():
        f = _front({"kv": None})
        f._asr_st()["waiters"]["pdflip-19-208"] = time.time()
        fits, why, _ = await Front._arrival_seat_kv(f, 64153, 64000, "pdflip-19-208", 0)
        assert fits is True and "kv=unread" in why

    asyncio.run(body())


def _gate(book, young, old, avail, need_of=None):
    return ds.admission_gate([young], running=[old], avail_tokens=avail, resume_book=book, need_of=need_of)


def _parked(rid, seq, n_in, out=0, site=None):
    r = types.SimpleNamespace(rid=rid, kv_arrival_seq=seq, origin_input_ids=[0] * n_in, output_ids=[0] * out,
                              is_fast_lane=False, spill_class=None, mamba_pool_idx=0,
                              sampling_params=SimpleNamespace(max_new_tokens=64000), finished=lambda: False)
    if site is not None:
        ds.mark_parked(r, site, epoch=7, now=100.0)
    return r


_ON = {"FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE": "1"}
_S = SimpleNamespace(page_size=1, token_to_kv_pool_allocator=_Alloc(0))


def test_i_resume_test_prices_like_the_gate_34_283():
    """34-283 (D TP0 23:16:40): PRESSURE-RESUME avail=108507 need=104951 (free+evictable vs. context); the gate in the
    same pass: price=109048 budget=107704.184 (offset=802) -- 568 refusals in 174.1 s. One formula: it stays parked."""
    old = _parked("pdflip-30-269", 269, 120000)
    young = _parked("pdflip-34-283", 283, 104704, out=247, site=ds.SITE_PRESSURE)
    assert dlr.resume_price(young, _S) == 109048                 # = D's price (extend 104951 + clip 4096 + page 1)
    budget = int(math.ceil(62131 + 46376 - 802)) - 1             # D's own terms: available + deliverable - offset
    assert budget == 107704
    book = ds.ResumeBook.from_env(_ON)
    got = [_gate(book, young, old, budget, need_of=lambda r: dlr.resume_price(r, _S)).skip(young)
           for _ in range(3 * book.steps)]
    assert set(got) == {"pdflip_d_park_older_live"}, "resumed into a gate that refuses it"
    book = ds.ResumeBook.from_env(_ON)                            # the base: context against free+evictable
    got = [_gate(book, young, old, 108507).skip(young) for _ in range(book.steps)]
    assert got[-1] is None


def test_r_resume_fires_when_the_gate_would_take_it():
    old = _parked("old", 1, 1000)
    young = _parked("y", 2, 60000, site=ds.SITE_PRESSURE)
    price = dlr.resume_price(young, _S)
    book = ds.ResumeBook.from_env(_ON)
    got = [_gate(book, young, old, price, need_of=lambda r: dlr.resume_price(r, _S)).skip(young)
           for _ in range(book.steps)]
    assert got[-1] is None                                       # budget == the largest admitted price: it resumes
    book = ds.ResumeBook.from_env(_ON)
    got = [_gate(book, young, old, price - 1, need_of=lambda r: dlr.resume_price(r, _S)).skip(young)
           for _ in range(book.steps)]
    assert got[-1] == "pdflip_d_park_older_live"


def test_j_resume_reading_is_the_min_of_each_ranks_own_lifetime_budget(monkeypatch):
    running = [_req("old", 116317, max_new=64000)]
    seen = []
    s0 = _sched(100000, _Tree(20000), running, ratio=0.2)
    s0._pdflip_group_min_ints = lambda vals: (seen.append(list(vals)), list(vals))[1]
    got = dpr._resume_fit_reading(s0, s0.running_batch)()
    own = int(math.ceil(120000 - CLIP * 0.2)) - 1
    assert seen == [[own, 1]] and got == own


def test_j2_the_resume_gate_uses_the_lifetime_price_when_armed(monkeypatch):
    import inspect

    src = inspect.getsource(dpr.admission)
    assert "need_of = _resume_need_of(sched)" in src and "need_of=need_of" in src
    src2 = inspect.getsource(ds.admission_gate)
    assert "int(need_of(r)) if need_of is not None" in src2


# ======================================================================== B: Verhungern
def test_k_an_old_d_without_the_fields_is_tested_as_before():
    rd = {"available": 40000, "evictable": 10000, "decode_clip": 4096}
    assert asr.kv_free(rd) == 50000
    assert asr.kv_fits(50000, rd) == (True, "kv need=50000 free=50000")
    assert asr.price_extra(rd) == 0


def test_l_the_kv_ladder_keeps_its_room_under_the_lifetime_budget():
    """NF: D grows its stage at the admission (D-MEM-SCHED) -- the stage's lifetime budget must not cap the ladder; the
    ladder room carries the same reserve and ledger."""
    rd = {"available": 10000, "evictable": 0, "ladder_ceiling": 524288, "ladder_used": 77000,
          "lifetime_budget": 9000, "decode_reserve": 2000, "ledger_outstanding": 500}
    assert asr.kv_free(rd) == 524288 - 77000 - 2000 - 500 - 1
    assert asr.kv_fits(400000, rd)[0] is True


def test_m_rows_d_already_holds_for_an_admitted_rid_are_not_charged_again():
    async def body():
        kv = {"available": 60000, "evictable": 0, "decode_clip": 4096, "running_rids": [],
              "admitted_rids": ["pdflip-9-61"], "displaceable_rids": []}
        f = _front({"kv": kv})
        await _seat(f, "pdflip-9-61", 40000)                     # unmerged extend on D: rows already in the reading
        fits, why, _ = await Front._arrival_seat_kv(f, 50000, 64000, "pdflip-9-62", 0)
        assert fits is True and "D-line debit" not in why

    asyncio.run(body())


def test_n_a_chunked_commitment_the_budget_subtracts_is_not_charged_again():
    async def body():
        kv = {"available": 60000, "evictable": 0, "decode_clip": 4096, "running_rids": [],
              "ledger_rids": ["pdflip-9-60"], "ledger_outstanding": 0, "displaceable_rids": []}
        f = _front({"kv": kv})
        await _seat(f, "pdflip-9-60", 40000)
        fits, why, _ = await Front._arrival_seat_kv(f, 50000, 64000, "pdflip-9-62", 0)
        assert fits is True, why

    asyncio.run(body())


def test_o_a_seat_is_charged_the_need_its_verdict_priced_not_its_prompt():
    async def body():
        kv = {"available": 70000, "evictable": 0, "decode_clip": 4096, "running_rids": ["prev"],
              "displaceable_rids": ["prev"]}
        f = _front({"kv": kv})
        await _seat(f, "prev", 63648)
        f.__dict__["_sess_prev"] = {"kurz": ("prev", 63648)}
        fits, why, _ = await Front._arrival_seat_kv(f, 64193, 64000, "kurz", 545)
        assert fits is True and "shared 63648" in why             # need 545 + 4096 = 4641
        f._asr_st()["granted"]["kurz"] = time.time()
        Front._asr_note_kv_priced(f, "kurz")
        await _seat(f, "kurz", 64193)
        fits, why, _ = await Front._arrival_seat_kv(f, 60000, 64000, "next", 0)
        assert "D-line debit 4641 of 1 seat(s)" in why and fits is True   # 70000 - 4641 >= 64096

    asyncio.run(body())


def test_p_s3_d_runs_none_of_the_seated_rids_the_head_is_never_held():
    async def body():
        kv = {"available": 1000, "evictable": 0, "decode_clip": 4096, "running_rids": [], "displaceable_rids": []}
        f = _front({"kv": kv})
        await _seat(f, "refused-on-d", 90000)
        fut = asyncio.get_running_loop().create_future()
        head = F.Pending("pdflip-30-274", "/generate", {"max_tokens": 64000}, "x", time.time(), fut,
                         est_prompt=58911, est_uncached=415, leg1_prompt_tokens=58911, d_eligible=True)
        f._ready_for_d.append(head)
        assert await Front._d_line_kv_holds(f, head) is False
        assert f.counters["d_admit_kv_nothing_runs"] == 1

    asyncio.run(body())


def test_q_a_displaced_decode_gives_its_reserve_back_in_the_age_plan():
    f = _front({"kv": None})
    now = time.time()
    f.__dict__.setdefault("_asr_arrive", {}).update({"head": now - 30.0, "young": now - 10.0})
    st = f._asr_st()

    class _S:
        rid, tokens = "young", 30000

    f._d_seats_live = [_S()]
    f.groups["D"].outstanding["young"] = now
    basis = {"unread": False, "wall": False, "running": {"young"}, "displaceable": {"young"},
             "reserve_by_rid": {"young": 2048}}
    assert Front._arrival_seat_age_verdict(f, "head", now - 30.0, True, 31000, basis=basis) == ["young"]
    basis["reserve_by_rid"] = {}
    st.get("age_plan", {}).clear()
    assert Front._arrival_seat_age_verdict(f, "head", now - 30.0, True, 31000, basis=basis) is None
