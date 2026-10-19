"""EZ-S1 (plan R.1, MEASURE_ON_D stage 1, review A3; SOLL R3/R7/R9/R11): measure and hold until the front decides.

Stage 1: D admits ``u < halbX`` (kurz, R2: right on D) and refuses every ``u >= halbX`` with its exact extent -- the
refusal IS the report; the front forms the class on THAT number. A half request is never prefilled on D without the
front's decision (M4: the barrier at X instead of halbX-1 lets it through).

* the barrier travels with the leg (``_front_x_headers``), once, and only to the request without an exact number;
* the report: gross/halb -> the set (R4/R3), under halbX -> an R7 DEFECT report (never a flip reason), a refusal without
  a parseable extent -> ``deferred``, an extent above u_p + one page -> ``mismatch`` (u_d <= u_p must hold);
* a failed SET-REPROBE member is a MEASURE request too;
* the front writes no ``D_NOW`` with ``u=0`` for such a request;
* switch off (``FLLIPER_PDFLIP_EXACT_CLASS=0``): ``_x_unpriced`` is the rc17 function byte for byte (golden below).
"""
import asyncio
import logging
import os
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("FLLIPER_PDFLIP_D_MATCH_PROBE_DEADLINE_MS", "150")
os.environ.setdefault("FLLIPER_PDFLIP_D_MATCH_PROBE_RETRY_MS", "10")

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import decision_numbers as dn  # noqa: E402
from flliper.srt.pdflip import decision_record as dr  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip import front_x  # noqa: E402
from flliper.srt.pdflip import phase_policy as pp  # noqa: E402
from test_pdflip_front_x_exact import X, _front  # noqa: E402
from test_decision_numbers_1019 import (HALF_X, N, PAGE, U_P, _R, _decisions, _drive, _img_payload, _mk,  # noqa: E402
                                        _parsed, _text_payload)


def _pend(rid="pdflip-0-1", est_prompt=N):
    p = F.Pending(rid, "/v1/messages", {}, "t", time.time(), asyncio.new_event_loop().create_future(),
                  est_prompt=est_prompt, est_uncached=0)
    return p


def _front_with_measure(rid="pdflip-0-1", u_p=U_P):
    f = _front(True)
    f._store_probe_info = {"page_size": PAGE}
    seq = F.Front._measure_start(f, rid, X, "test", u_p=u_p)
    return f, seq


def _send_header(f, rid):
    return F.Front._front_x_headers(f, rid, None, False)


def _report(f, rid, extent, p=None):
    body = b'{"error": "W50: this request\'s extent after prefix matching is %d. Re-routed"}' % extent
    F.Front._measure_return(f, rid, p or _pend(rid), F._d_refusal_extent(body), N)


# -------------------------------------------------------------------------------------------- the barrier travels

def test_the_barrier_rides_with_the_leg_once_and_only_for_the_measure_request(caplog):
    f, seq = _front_with_measure()
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        hdr = _send_header(f, "pdflip-0-1")
        again = _send_header(f, "pdflip-0-1")
    assert hdr[front_x.HEADER_ACTION] == "MEASURE_ON_D" and hdr[front_x.HEADER_HALFX] == str(HALF_X)
    assert int(hdr[front_x.HEADER_X]) + PAGE == HALF_X - 1 and hdr[front_x.HEADER_PRICED] == "0"
    assert front_x.HEADER_ACTION not in again, "the marker is spent with the leg that carried it"
    other = _send_header(f, "pdflip-0-2")
    assert front_x.HEADER_ACTION not in other
    assert any("action=MEASURE_ON_D" in m and "halfX-1 = %d" % (HALF_X - 1) in m for m in caplog.messages)
    done = [dr.parse_line(m) for m in caplog.messages if m.startswith("WEG2 DECISION-DONE")]
    assert done and done[0]["seq"] == seq and done[0]["executor"] == "front"


def test_the_barrier_is_dropped_for_a_hand_over_after_p():
    f, _ = _front_with_measure()
    p = _pend()
    p.d_direct = False                    # leg 1 ran on P: D reads P's KV back, prefills ~nothing
    hdr = F.Front._front_x_headers(f, "pdflip-0-1", p, False)
    assert front_x.HEADER_ACTION not in hdr and not f.__dict__.get("_measure_on_d", {}).get("pdflip-0-1")


def test_the_switch_off_sends_no_barrier():
    with envs.FLLIPER_PDFLIP_EXACT_CLASS.override(False):
        f, _ = _front_with_measure()
        assert front_x.HEADER_ACTION not in _send_header(f, "pdflip-0-1")


def test_front_x_carry_off_means_no_header_at_all():
    with envs.FLLIPER_PDFLIP_FRONT_X_CARRY.override(False):
        f, _ = _front_with_measure()
        assert _send_header(f, "pdflip-0-1") == {}


def test_the_action_header_is_stamped_on_the_request_for_d():
    class H(dict):
        pass
    adapted = SimpleNamespace()
    raw = SimpleNamespace(headers={front_x.HEADER_X: "1983", front_x.HEADER_PRICED: "0",
                                   front_x.HEADER_HALFX: "2048", front_x.HEADER_ACTION: "MEASURE_ON_D"})
    front_x.stamp_request(adapted, raw)
    assert getattr(adapted, front_x.ATTR_ACTION) == "MEASURE_ON_D" and getattr(adapted, front_x.ATTR_X) == 1983


# -------------------------------------------------------------------------------------------- the report

def test_report_gross_goes_to_the_set_under_r4(caplog):
    f, seq = _front_with_measure()
    _send_header(f, "pdflip-0-1")
    p = _pend()
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        _report(f, "pdflip-0-1", U_P, p)
    d = _parsed(caplog)
    assert [(x["kind"], x["action"], x["rule"], x["classes"]) for x in d] == [("measure", "COLLECT", "R4", ["gross"])]
    assert p.measured_u == U_P and p.measure_src == "d_extent"
    done = _parsed(caplog, "DECISION-DONE")
    assert [(x["seq"], x["outcome"], x["measured_u"], x["src"]) for x in done] == [(seq, "ok", U_P, "d_extent")]


def test_report_halb_goes_to_the_set_under_r3(caplog):
    f, _ = _front_with_measure()
    _send_header(f, "pdflip-0-1")
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        _report(f, "pdflip-0-1", HALF_X + 10, _pend(est_prompt=HALF_X + 100))
    d = _parsed(caplog)
    assert [(x["action"], x["rule"], x["classes"]) for x in d] == [("COLLECT", "R3", ["halb"])]


def test_report_under_halfx_is_an_r7_defect_not_a_flip_reason(caplog):
    f, _ = _front_with_measure()
    _send_header(f, "pdflip-0-1")
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        _report(f, "pdflip-0-1", HALF_X - 1, _pend(est_prompt=5000))
    d = _parsed(caplog)
    assert [(x["action"], x["rule"], x["classes"]) for x in d] == [("TO_D", "R7", ["kurz"])]
    assert "d_refused_under_x" in d[0]["why"]
    done = _parsed(caplog, "DECISION-DONE")
    assert done[0]["outcome"] == "mismatch"


def test_report_without_an_extent_is_deferred_and_forms_no_class(caplog):
    f, seq = _front_with_measure()
    _send_header(f, "pdflip-0-1")
    p = _pend()
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        F.Front._measure_return(f, "pdflip-0-1", p, None, N)
    assert not _parsed(caplog) and p.measured_u is None
    done = _parsed(caplog, "DECISION-DONE")
    assert [(x["seq"], x["outcome"]) for x in done] == [(seq, "deferred")]


def test_report_of_an_extent_above_u_p_plus_a_page_is_a_mismatch(caplog):
    f, _ = _front_with_measure(u_p=5000)          # u_d <= u_p must hold; D says 9000
    _send_header(f, "pdflip-0-1")
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        _report(f, "pdflip-0-1", 9000, _pend(est_prompt=20000))
    done = _parsed(caplog, "DECISION-DONE")
    assert done[0]["outcome"] == "mismatch" and "u_p=5000" in done[0]["detail"]


def test_an_ordinary_w31_refusal_is_not_a_measure_report(caplog):
    f = _front(True)
    p = _pend("pdflip-0-9")
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        F.Front._measure_return(f, "pdflip-0-9", p, 12000, N)
    assert not _parsed(caplog) and p.measured_u is None


def test_the_class_is_formed_at_the_x_of_the_exactly_known_depth(monkeypatch):
    """After D's report the depth is exact (n - extent): X is read at THAT depth, not at the arrival's guess."""
    f, _ = _front_with_measure()
    _send_header(f, "pdflip-0-1")
    seen = []
    orig = F.Front._x_at_depth
    monkeypatch.setattr(F.Front, "_x_at_depth", lambda self, depth, k=1: (seen.append(depth) or orig(self, depth, k)))
    p = _pend(est_prompt=N)
    F.Front._measure_return(f, "pdflip-0-1", p, U_P, N)
    assert seen == [N - U_P]


# -------------------------------------------------------------------------------------------- other doors

def test_no_d_now_with_a_zero_member_for_a_measure_request(monkeypatch, caplog):
    f = _mk(monkeypatch)
    # the REAL seat path (not the stub of the e2e harness): its D_NOW line must not name this request
    seats = {}

    async def real_core(rid, est_tokens, refused, max_tokens, uncached, x_set, t_in):
        seats["n"] = seats.get("n", 0) + 1
        return object()
    f._acquire_short_seat_core = real_core
    f._acquire_short_seat = lambda *a, **k: F.Front._acquire_short_seat(f, *a, **k)
    f._set_seats_free = lambda: 3
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        _drive(f, _text_payload())
    msgs = caplog.messages
    assert seats.get("n") == 1
    assert not [m for m in msgs if m.startswith("PDFLIP SET-VERDICT") and "verdict=D_NOW" in m]
    assert [m for m in msgs if "kind=measure action=MEASURE_ON_D" in m]


def test_a_failed_reprobe_member_becomes_a_measure_request(caplog):
    f = _front(True)
    f._store_probe_info = {"page_size": PAGE}
    p = _pend("pdflip-3-5", est_prompt=9000)
    p.est_uncached = 7000
    f.queue.append(p)
    f._set_depth = {"pdflip-3-5": 2000}
    e = SimpleNamespace(t=time.monotonic())
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        route = f._set_reprobe_unpriced("pdflip-3-5", e, "no exact D number", "set_verdict")
    assert "d_admission" in route
    assert p.est_uncached == 0
    d = _parsed(caplog)
    assert [(x["kind"], x["action"]) for x in d if x["kind"] == "measure"] == [("measure", "MEASURE_ON_D")]
    hdr = _send_header(f, "pdflip-3-5")
    assert hdr[front_x.HEADER_ACTION] == "MEASURE_ON_D"


# -------------------------------------------------------------------------------------------- switch-off golden (rc17)

def _rc17_x_unpriced(self, rid, c, est_prompt, uncached_p, media):
    """The rc17 ``Front._x_unpriced`` (b52b4b9b53), verbatim apart from the logging."""
    n = int(c.n) if (c is not None and not media) else int(est_prompt)
    return dict(pending=0, n=n, known=False, credit=0, src="d_probe_unpriced", store_depth=0, credit_epoch=None,
                mm=False, mm_cached=False, image_end=0, d_reach=0, d_floor=0, probe=None, image_in_tail=False,
                unpriced=True, uncached_p=int(uncached_p),
                n_src="front_ids" if (c is not None and not media) else "size_only")


@pytest.mark.parametrize("media", [False, True])
@pytest.mark.parametrize("has_c", [False, True])
@pytest.mark.parametrize("up", [-1, 0, 202, HALF_X - 1, HALF_X, U_P])
def test_switch_off_x_unpriced_is_the_rc17_function(media, has_c, up):
    with envs.FLLIPER_PDFLIP_EXACT_CLASS.override(False):
        f = _front(True)
        c = SimpleNamespace(n=N) if has_c else None
        got = F.Front._x_unpriced(f, "r", c, 777, up, media)
        want = _rc17_x_unpriced(f, "r", c, 777, up, media)
        for k, v in want.items():
            assert getattr(got, k) == v, k
        assert got.measure is False and got.depth_hint is None and got.class_basis is None


@pytest.mark.parametrize("media,has_c,up,kind", [
    (False, True, 202, dn.KIND_P_UPPER), (False, True, HALF_X - 1, dn.KIND_P_UPPER),
    (False, True, HALF_X, dn.KIND_MEASURE), (False, True, U_P, dn.KIND_MEASURE), (False, True, -1, dn.KIND_MEASURE),
    (True, False, 202, dn.KIND_MEASURE), (True, True, 202, dn.KIND_MEASURE), (False, False, 202, dn.KIND_MEASURE)])
def test_switch_on_x_unpriced_follows_the_ladder(media, has_c, up, kind):
    f = _front(True)
    c = SimpleNamespace(n=N) if has_c else None
    got = F.Front._x_unpriced(f, "r", c, 777, up, media)
    assert got.class_basis.kind == kind and got.measure == (kind == dn.KIND_MEASURE)
    if kind == dn.KIND_P_UPPER:
        assert got.pending == up and got.depth_hint == N - up
    else:
        assert got.pending == 0, "pending 0 is only the ROUTE to D's admission, never a price"
    assert got.n_src == ("front_ids" if (has_c and not media) else "n_est")


def test_the_switch_off_e2e_sends_no_measure_and_writes_no_measure_decision(monkeypatch, caplog):
    with envs.FLLIPER_PDFLIP_EXACT_CLASS.override(False):
        f = _mk(monkeypatch)
        with caplog.at_level(logging.INFO, logger="pdflip.front"):
            _drive(f, _text_payload())
        assert not [d for d in _parsed(caplog) if d["kind"] == "measure"]
        assert f.routed and not f.__dict__.get("_measure_on_d")
        hdr = F.Front._front_x_headers(f, f.routed[0][1], None, False)
        assert front_x.HEADER_ACTION not in hdr
