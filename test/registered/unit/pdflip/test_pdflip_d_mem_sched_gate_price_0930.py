"""Flipzeit-Regression 30.09. (NF y3z, boot 0930_023308, ep44 and ep58): the
wake admitted every held request but one, the one got the attention host's
NO_TOKEN at a FREE seat, and D-MEM-SCHED stayed on its stage ("holds") --
the stage ladder reached S9 in the same boot (S7 at 02:52:15, 38 s earlier).

D-MEM-SCHED sized the stage for ``used + incoming + air``; the admission gate
(``PrefillAdder.add_one_req``) charges more: every running request's decode
reservation (``min(max_new - out, CLIP) * new_token_ratio``, the adder's
``rem_total_token_offset``) and the candidate's own ``min(max_new, CLIP) +
page``. The machine saw room the gate did not give:

* ep44 02:53:25, S6 = 229376: need 144571 + 78477 + 4120 = 227168 (holds);
  gate price 82637 > budget 78336 -- pdflip-42-101 waited for pdflip-41-100's
  end until 02:53:29 (flip 7,32 s, D-Nachlauf 5,14 s).
* ep58 02:58:12, S5 = 196608, bs=4 of 6: need ~193.9k (holds); price 63071
  > budget 59392 -- pdflip-56-153 waited 8,1 s (D-Nachlauf 9,0 s).

The fix prices the queue like the gate (only while something is queued: a
reservation never grows the stage on its own), so the stage grows and
``_reopen_admission`` clears the NO_TOKEN's batch-full gate in the same tick.
"""

import logging
import types

import pytest

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GRID = [32768, 65536, 98304, 131072, 163840, 196608, 229376, 262144, 393216, 524288]
CLIP = 4096


def _req(rid, n_in, n_out=0, max_new=32768):
    return types.SimpleNamespace(
        rid=rid, origin_input_ids=[0] * n_in, output_ids=[0] * n_out, finished=lambda: False,
        sampling_params=types.SimpleNamespace(max_new_tokens=max_new))


@pytest.fixture
def env(monkeypatch):
    from flliper.srt.pdflip import d_seat_vram as dsv

    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.setenv("FLLIPER_OPT_PDFLIP_D_SEAT_VRAM", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_D_KV_STAGE_TOKENS", ",".join(str(t) for t in GRID))
    monkeypatch.setenv("FLLIPER_PDFLIP_D_KV_STAGE_BY_DEMAND", "1")
    monkeypatch.setenv("FLLIPER_CLIP_MAX_NEW_TOKENS_ESTIMATION", str(CLIP))
    caps = []
    monkeypatch.setattr(dsv, "_engage_kv_cap", lambda alloc, t, p: caps.append(int(t)) or t)
    monkeypatch.setattr(dsv, "max_live_page", lambda alloc: 0)

    def make(stage, ratio):
        sched = types.SimpleNamespace(
            # the y3z D argv: chunk 4096, 6 seats, page 1; air = 4096 + 6 x 4 = 4120
            server_args=types.SimpleNamespace(max_running_requests=6, chunked_prefill_size=4096,
                                              speculative_num_draft_tokens=4, page_size=1),
            running_batch=types.SimpleNamespace(reqs=[], batch_is_full=False), waiting_queue=[],
            chunked_req=None, last_batch=None, page_size=1, token_to_kv_pool_allocator=object(),
            new_token_ratio_tracker=types.SimpleNamespace(current=ratio),
            _pdflip_group_min_ints=lambda vals: vals)
        setattr(sched, dsv.CTL_ATTR, False)
        setattr(sched, dsv.PHASE_ATTR, dsv.PhaseState(epoch="e44", n=6, cap=6, done=True,
                                                      stage=stage, stage_tokens=GRID[stage]))
        return sched

    return dsv, make, caps


def _ep44(make):
    """02:53:25: 4 x 16.6k + pdflip-41-100 running (used 144571), pdflip-42-101
    (78477) queued behind the host's NO_TOKEN, one seat free, stage S6.
    ratio 0.316: the host's budget 78336 = 229376 - 144571 - 6469 (5 x 4096 x
    0.316 = 6472)."""
    sched = make(6, 0.316)
    run = [_req("pdflip-40-%d" % i, 16448, 100) for i in (96, 97, 98, 99)]
    run.append(_req("pdflip-41-100", 78144, 235))
    sched.running_batch.reqs = run
    sched.running_batch.batch_is_full = True          # the NO_TOKEN of 02:53:26
    sched.waiting_queue = [_req("pdflip-42-101", 78477)]
    return sched


def test_ep44_the_gate_price_grows_s6_to_s7_and_reopens_the_admission(env, caplog):
    dsv, make, _caps = env
    sched = _ep44(make)
    used, incoming, _ = dsv.global_demand(sched)
    assert (used, incoming) == (144571, 78477)
    assert used + incoming + 4120 <= GRID[6]          # the census alone: S6 "holds"
    with caplog.at_level(logging.INFO):
        dsv.runtime_tick(sched)
    st = getattr(sched, dsv.PHASE_ATTR)
    # gate: 144571 used + 6472 decode reservations + 78477 + 4096 + 1 = 233617 > S6
    assert st.stage_tokens == GRID[7]                 # base: 229376 (S6 held)
    assert sched.running_batch.batch_is_full is False # base: True until pdflip-41-100 ended
    assert any(dsv.REOPEN_MARK in r.getMessage() for r in caplog.records)


def test_ep58_two_free_seats_s5_to_s6(env):
    """02:58:12, bs=4 of 6: pdflip-46-11 (81664) + 3 x 16384 running, pdflip-56-153
    (58975) refused at S5 (price 63071 > budget 59392)."""
    dsv, make, _caps = env
    sched = make(5, 0.39)
    sched.running_batch.reqs = [_req("pdflip-46-11", 81664, 40)] + [
        _req("pdflip-52-%d" % i, 16384, 40) for i in (149, 150, 151)]
    sched.running_batch.batch_is_full = True
    sched.waiting_queue = [_req("pdflip-56-153", 58975)]
    used, incoming, _ = dsv.global_demand(sched)
    assert used + incoming + 4120 <= GRID[5]          # the census alone: S5 "holds"
    dsv.runtime_tick(sched)
    assert getattr(sched, dsv.PHASE_ATTR).stage_tokens == GRID[6]   # base: 196608
    assert sched.running_batch.batch_is_full is False


def test_the_reservation_alone_never_grows_the_stage(env):
    """No queue: the running requests' decode reservations are the gate's
    price for an ADMISSION only -- no expert row leaves for them."""
    dsv, make, _caps = env
    sched = _ep44(make)
    sched.waiting_queue = []
    sched.running_batch.batch_is_full = False
    step = dsv.runtime_tick(sched)
    assert getattr(sched, dsv.PHASE_ATTR).stage_tokens <= GRID[6]
    assert step is None or not step.changed or step.stage <= 6


def test_the_gate_price_is_the_adders_price(env):
    """The reservation follows ``PrefillAdder``: running ``min(max_new - out,
    CLIP) * new_token_ratio`` + candidate ``min(max_new - out, CLIP) + page``;
    a request without max_new (a test double, a warmup) reserves nothing."""
    dsv, make, _caps = env
    sched = make(0, 0.5)
    running = [_req("a", 100, 10, max_new=110), _req("b", 100, 0, max_new=100000)]
    queued = [_req("q", 5000, 0, max_new=1000), types.SimpleNamespace(rid="x", origin_input_ids=[0] * 7)]
    # a: min(100, 4096) * 0.5 = 50; b: 4096 * 0.5 = 2048; q: 1000 + 1; x: 0 + 1
    assert dsv.admission_reserve(sched, running, queued) == 50 + 2048 + 1001 + 1
    assert dsv.admission_reserve(sched, running, []) == 0
