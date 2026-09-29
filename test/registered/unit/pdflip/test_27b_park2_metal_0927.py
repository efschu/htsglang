"""27B park, second round (PK2) -- the two findings of the first metal runs of
profile 27b-park-draft (rc12g-flat, 27.09.2026). Each test is red on unified
3e97ef0c8f and green with the fix.

(b) boot dkr27bparkdraftbar1w209270712, epoch 28 -> flip D->P of 67.43 s, 65.7 s
    of it the drain: two BAND requests (pdflip-28-62 uncached 4172, pdflip-28-65 4370;
    X_busy 4096 < u <= live X 7290) were queued by the RC7-X busy/idle split
    (Pending.x_deferred) while D decoded. The immediate trigger compared them with
    the live X and never fired; 45 s later the FAIRNESS switch closed admission and
    the flip DRAINED two agent decodes (6198 / 11133 tokens) instead of parking.
    Fix 1: x_deferred fires (the X in force while D decodes is X_busy).
    Fix 2: under the immediate park every D->P flip with running decodes parks
    first (cause=before-flip), whatever decided the flip.
(a) boot dkr27bparkdraftbar1w209270645, park probe: B-TTFT 26.8-27.1 s = 6.3 s
    (park + flip + P prefill 2.5 s + flip) + 20.1 s on D: A's read after the wake
    answered ZERO (#1478 shape, no #1324 stamp) -> #1471 settle held A for the
    whole 20 s bound, and the park's admission barrier held the fully loaded B
    behind A. Then A was prefilled from 0 (418 tokens) anyway.
    Fix 3: under the immediate park alone a settle-held parked request does not
    bar newcomers. Fix 4: a zero answer settles when the whole span fits in X.
    (A finish=length is the probe's own max_tokens=3000, not a defect.)
Hermetic, CPU only.
"""
from __future__ import annotations

import asyncio
import importlib.util
import os
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import d_park_runtime as rt  # noqa: E402
from flliper.srt.pdflip import d_seats as ds  # noqa: E402
from flliper.srt.pdflip import form as FM  # noqa: E402
from flliper.srt.pdflip import phase_policy as pp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
IMM = "FLLIPER_PDFLIP_D_PARK_IMMEDIATE"


def _form_env(profile):
    arch, experts, draft, kv = (("dense", "none", "dflash", "paged_dcp") if profile == "qwen27b"
                                else ("moe", "offload", "mtp", "qsa_forma"))
    return FM.PdFlipForm(arch=arch, experts=experts, draft=draft, p_draft="none", kv=kv,
                       flip="family", vision="off", profile=profile, model="m").env_value()


@pytest.fixture
def clean(monkeypatch):
    for k in (IMM, "FLLIPER_PDFLIP_STANDARD_FORM", "FLLIPER_PDFLIP_D_PARK", FM.FORM_ENV,
              "FLLIPER_PDFLIP_GROUP", "FLLIPER_PDFLIP_STORE_SHORT_TAIL", "FLLIPER_PDFLIP_STORE_SHORT_TAIL_X"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


@pytest.fixture
def d27(clean):
    clean.setenv(FM.FORM_ENV, _form_env("qwen27b"))
    clean.setenv("FLLIPER_PDFLIP_GROUP", "D")
    clean.setenv(IMM, "1")
    return clean


# ---------------------------------------------------------------- (b) fix 1
def test_a_band_deferred_request_fires_the_immediate_park():
    """pdflip-28-62: est_uncached 4172, live X 7290, X_busy 4096, x_deferred."""
    band = types.SimpleNamespace(rid="pdflip-28-62", est_uncached=4172, skip_leg1=False,
                                 leg1_done=False, p_only=False, x_requeues=0, x_deferred=True)
    assert pp.needs_p(4172, 7290, x_deferred=True)
    assert pp.immediate_park_trigger([band], 7290) is band
    band.x_deferred = False  # a SHORT D-direct candidate of the same size: no flip due
    assert pp.immediate_park_trigger([band], 7290) is None
    band.x_deferred, band.leg1_done = True, True  # prefilled already: waits for D, not P
    assert pp.immediate_park_trigger([band], 7290) is None


# ---------------------------------------------------------------- (b) fix 2
def _h91c():
    spec = importlib.util.spec_from_file_location(
        "_park2_h91c", os.path.join(HERE, "test_pdflip_phase_policy_h91c.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_fairness_flip_under_the_immediate_park_parks_instead_of_draining(clean, monkeypatch):
    """The trigger does not see r1 (as for the band requests on the base); the
    fairness bound fires. Base: the flip drains r0 (held) -> no flip while r0
    decodes. Fix: park (cause=before-flip), then the flip; r0 resumes after."""
    clean.setenv(FM.FORM_ENV, _form_env("qwen27b"))
    H = _h91c()
    monkeypatch.setattr(pp, "immediate_park_trigger", lambda queue, x: None)

    async def body():
        async with H.Harness(awake="P", p_concurrency=4, d_bs=2, d_park_immediate=True,
                             w_s=1.0) as h:
            h.d.hold = {}
            t0 = h.post("r0")
            assert await H._until(lambda: h.d.running, 20)
            t1 = h.post("r1")
            assert await H._until(lambda: "gen:r1" in h.p.timeline, 15), h.d.timeline
            assert [b["reason"] for b in h.d.park_bodies] == ["immediate-over-x"]
            tl = h.d.timeline
            i_gen, i_park = H._first(tl, "gen:r0"), H._first(tl, "rpc:pdflip/park_running")
            rel = [i for i, e in enumerate(tl) if e == "rpc:release_memory_occupation"]
            assert i_gen < i_park and not [i for i in rel if i_gen < i < i_park]
            assert [i for i in rel if i > i_park]
            assert h.front.counters["park_immediate_before-flip"] == 1
            assert not t0.done()
            assert await H._until(lambda: h.front.awake == "D" and not h.front._d_parked, 20)
            h.d.release_all()
            (s0, _), (s1, _) = await asyncio.wait_for(asyncio.gather(t0, t1), 20)
            assert (s0, s1) == (200, 200) and h.d.gen_marks.count("r0") == 1

    asyncio.run(body())


def test_switch_off_a_fairness_flip_still_drains_as_before(clean, monkeypatch):
    clean.setenv(FM.FORM_ENV, _form_env("qwen27b"))
    H = _h91c()

    async def body():
        async with H.Harness(awake="P", p_concurrency=4, d_bs=2, w_s=1.0) as h:
            h.d.hold = {}
            t0 = h.post("r0")
            assert await H._until(lambda: h.d.running, 20)
            t1 = h.post("r1")
            await asyncio.sleep(2.5)
            assert h.d.park_bodies == [] and "rpc:release_memory_occupation" not in h.d.timeline[
                H._first(h.d.timeline, "gen:r0"):]
            h.d.release_all()
            (s0, _), (s1, _) = await asyncio.wait_for(asyncio.gather(t0, t1), 20)
            assert (s0, s1) == (200, 200)

    asyncio.run(body())


# ---------------------------------------------------------------- (a) fix 3
def _req(rid, seq):
    return types.SimpleNamespace(rid=rid, kv_arrival_seq=seq, origin_input_ids=[0] * 10,
                                 output_ids=[0] * 3, is_fast_lane=False, spill_class=None)


def _sched(waiting, settle):
    return types.SimpleNamespace(waiting_queue=list(waiting), pdflip_post_wake_settle=list(settle),
                                 pdflip_dormant_hold=[], uniform_min_avail=lambda: 0)


def test_a_settle_held_parked_request_does_not_bar_the_newcomer(d27):
    a, b = _req("A-parked", 1), _req("B-new", 2)
    ds.mark_parked(a, ds.SITE_FLIP)
    s = _sched([b], [a])
    gate = rt.admission(s, types.SimpleNamespace(reqs=[]))
    assert gate is None or gate.skip(b) is None


def test_a_parked_request_in_the_queue_still_goes_first(d27):
    a, b = _req("A-parked", 1), _req("B-new", 2)
    ds.mark_parked(a, ds.SITE_FLIP)
    s = _sched([b, a], [])
    gate = rt.admission(s, types.SimpleNamespace(reqs=[]))
    assert gate is not None and gate.skip(b) == "pdflip_d_park_first" and gate.skip(a) is None


def test_nf_standard_form_keeps_the_settle_in_the_barrier(clean):
    clean.setenv(FM.FORM_ENV, _form_env("nextflash"))
    clean.setenv("FLLIPER_PDFLIP_GROUP", "D")
    a, b = _req("A-parked", 1), _req("B-new", 2)
    ds.mark_parked(a, ds.SITE_FLIP)
    gate = rt.admission(_sched([b], [a]), types.SimpleNamespace(reqs=[]))
    assert gate is not None and gate.skip(b) == "pdflip_d_park_first"


# ---------------------------------------------------------------- (a) fix 4
def _tail_req(n_ids, short=True, delivered=None):
    r = types.SimpleNamespace(rid="A", full_untruncated_fill_ids=list(range(n_ids)), _1471_short=short)
    if delivered is not None:
        r._pdflip_store_delivered = delivered
    return r


@pytest.mark.parametrize("n,short,delivered,want", [
    (418, True, None, True),      # metal A: zero answer, whole span 418 <= X -> D prefills now
    (98210, True, None, False),   # zero answer over X: waits for its read (weg2xsn229 unchanged)
    (418, False, None, False),    # no short mark, no stamp: not this case
    (4316, True, 4095, True),     # stamped short read, remainder 221 (weg2rc2, unchanged)
])
def test_a_zero_answer_settles_when_the_whole_span_fits_in_x(clean, n, short, delivered, want):
    from flliper.srt.managers import scheduler as S

    clean.setenv(FM.FORM_ENV, _form_env("qwen27b"))  # store_short_tail on (profile)
    clean.setenv(IMM, "1")  # the zero-read settle rides the 27B immediate park
    sched = types.SimpleNamespace(server_args=types.SimpleNamespace(tp_prefill_max_tokens=4096))
    assert S._pdflip_store_tail_settles(sched, _tail_req(n, short, delivered)) is want


@pytest.mark.parametrize("profile", ["qwen27b", "nextflash"])
def test_the_row_default_carries_the_zero_read_settle(clean, profile):
    # 28.09.: the immediate park is the registry default of both rows now, so the
    # zero-read settle that rides it is on without an explicit switch.
    from flliper.srt.managers import scheduler as S

    clean.setenv(FM.FORM_ENV, _form_env(profile))
    clean.setenv("FLLIPER_PDFLIP_STORE_SHORT_TAIL", "1")
    sched = types.SimpleNamespace(server_args=types.SimpleNamespace(tp_prefill_max_tokens=4096))
    assert S._pdflip_store_tail_settles(sched, _tail_req(418, True, None)) is True


@pytest.mark.parametrize("profile,imm", [("nextflash", "0"), ("qwen27b", "0")])
def test_without_the_immediate_park_a_zero_answer_waits_as_before_pk2(clean, profile, imm):
    # NF 27.09. (Fork A on #1004): PK2 settled zero reads on NF-D against its
    # 12288 riegel. Off the immediate park a zero read waits out #1471 again;
    # the stamped #1324 tail (remainder within X) is untouched.
    from flliper.srt.managers import scheduler as S

    clean.setenv(FM.FORM_ENV, _form_env(profile))
    clean.setenv("FLLIPER_PDFLIP_STORE_SHORT_TAIL", "1")
    if imm is not None:
        clean.setenv(IMM, imm)
    sched = types.SimpleNamespace(server_args=types.SimpleNamespace(tp_prefill_max_tokens=12288))
    assert S._pdflip_store_tail_settles(sched, _tail_req(418, True, None)) is False
    assert S._pdflip_store_tail_settles(sched, _tail_req(4316, True, 4095)) is True
