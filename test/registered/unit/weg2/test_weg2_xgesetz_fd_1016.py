"""X-GESETZ F-D (user 16.10.: "aktuell wird in P DEUTLICH unter der X-Schwelle
geprefillt"; root analysis deskq/done/nf-xgesetz-short-auf-p-1016.md, class D).

NF h16s @ 1f4b3e73d4, boot 1016_124240, the chain of weg2-6-55:

  front 12:51:11,031  X-ROUTE rid=weg2-6-55 est_uncached=375 X=6682 (EXACT)
  front 12:51:36,823  WEG2-ROUTE rid=weg2-6-55 SHORT -> D ... remainder=375 presence_span=76544
  front 12:55:13,658  PARK-HANDBACK rid=weg2-6-55 kept=committed (... D prefills it after the flip)
  D     12:55:46      W50-REROUTE rid=weg2-6-55 d_extent=375 X=12288 reason=store_short_after_flip
  D     12:55:46      D-NORECOMPUTE rid=weg2-6-55 would_compute=375 -> RESUME-VIA-P
  front 12:56:01,657  PARK-IMMEDIATE FIRED cause=over-x rid=weg2-6-55 est_uncached=375 p_only=True
                      -- a D->P flip of its own for a 375-token SHORT, six decodes parked.

Law 4 lets D prefill under X; D-NORECOMPUTE (02.10.) is for the hand-over after a P
prefill. A D-direct leg (front X on the request, no output yet) within that X is D's
own extend and is not diverted.
"""
from __future__ import annotations

import inspect
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

X_6_55 = 6682      # the front's decision X at the SHORT -> D dispatch
N_6_55 = 76919     # the context P would have prefilled (W50-REROUTE line)
DELIV_6_55 = 76544  # 76919 - 375: the wake read's synced witness


def _sched(static_x=12288, page=64):
    return types.SimpleNamespace(server_args=types.SimpleNamespace(tp_prefill_max_tokens=static_x),
                                 page_size=page, ps=types.SimpleNamespace(tp_rank=0), weg2_d_parked=[])


def _req(rid="weg2-6-55", n=N_6_55, delivered=DELIV_6_55, front_x=X_6_55, priced=375, output_ids=None):
    return types.SimpleNamespace(rid=rid, full_untruncated_fill_ids=list(range(n)),
                                 _weg2_store_delivered=delivered, stream=True, multimodal_inputs=None,
                                 origin_input_ids=list(range(n)), output_ids=list(output_ids or []),
                                 weg2_front_x=front_x, weg2_front_priced=priced)


def _release_filter(sched, release, dnr, diverted):
    """The scheduler's settle-release gate, statement for statement (the source
    check below pins the scheduler to this same shape)."""
    out = []
    for r in release:
        rem = dnr.d_would_compute(r)
        if rem > 0 and not dnr.d_direct_extends(sched, r, rem) and diverted(sched, r, rem):
            continue
        out.append(r)
    return out


def test_fd_weg2_6_55_is_extended_on_d_not_diverted(monkeypatch, caplog):
    import logging

    from sglang.srt.weg2 import d_norecompute as dn

    caplog.set_level(logging.INFO)
    calls = []

    def _div(sched, r, rem):
        calls.append((r.rid, rem))
        return True

    req = _req()
    assert dn.d_would_compute(req) == 375
    kept = _release_filter(_sched(), [req], dn, _div)
    assert kept == [req], "the D-direct SHORT stays in D's release (law 4), no RESUME-VIA-P"
    assert calls == [], "no D->P divert for a D-direct leg under its X"
    assert any("D-DIRECT-EXTEND rid=weg2-6-55 would_compute=375 front_x=6682 limit=6746" in m
               for m in caplog.messages)


def test_fd_handover_after_p_prefill_diverts_only_a_tail_of_halb_static_x_or_more():
    """No front X on the leg = D reads it back from P's hand-over. H16V F3 (SOLL R2/R7): a tail under half the static
    X is D's own kurz work (was: 201 tokens -> RESUME-VIA-P -> a D->P flip); a tail from half the static X up still
    goes through D-NORECOMPUTE."""
    from sglang.srt.weg2 import d_norecompute as dn

    calls = []
    req = _req(rid="weg2-16-79", n=54345, delivered=54144, front_x=None, priced=None)
    kept = _release_filter(_sched(), [req], dn, lambda s, r, rem: calls.append(rem) or True)
    assert kept == [req] and calls == []
    big = _req(rid="weg2-big", n=54345, delivered=54345 - 6144, front_x=None, priced=None)
    kept = _release_filter(_sched(), [big], dn, lambda s, r, rem: calls.append(rem) or True)
    assert kept == [] and calls == [6144]


def test_fd_resumed_request_tail_is_bounded_by_half_of_its_carried_front_x():
    """H16V F3 (SOLL R2/R7): a request that already decoded keeps its carried front X (``weg2_front_x``); a tail
    under halbX of it is D's own extend (NF h16u2 weg2-18-250: d_extent 77 flipped the whole group), a tail of
    halbX or more is RESUME-VIA-P as before."""
    from sglang.srt.weg2 import d_norecompute as dn

    req = _req(output_ids=[1, 2, 3])
    assert dn.d_direct_extends(_sched(), req, 375) is True
    assert dn.d_direct_extends(_sched(), req, 3340) is True      # 3340 < 0.5 * 6682
    assert dn.d_direct_extends(_sched(), req, 3341) is False     # halbX itself is not kurz
    assert dn.d_direct_extends(_sched(static_x=2000), req, 2001) is False   # the static X still bounds it


@pytest.mark.parametrize("rem,expect", [(6682 + 64, True), (6682 + 65, False), (12000, False)])
def test_fd_limit_is_the_x_gates_limit(rem, expect):
    """The bound is exactly D's X gate limit min(static X, front X + one page)."""
    from sglang.srt.weg2 import d_norecompute as dn

    assert dn.d_direct_extends(_sched(), _req(), rem) is expect


def test_fd_static_x_caps_the_front_x():
    from sglang.srt.weg2 import d_norecompute as dn

    req = _req(front_x=20000)
    assert dn.d_direct_extends(_sched(static_x=12288), req, 12288) is True
    assert dn.d_direct_extends(_sched(static_x=12288), req, 12289) is False


def test_fd_scheduler_gate_asks_before_divert():
    from sglang.srt.managers import scheduler as S

    src = inspect.getsource(S.Scheduler._weg2_post_wake_settle_tick)
    i = src.index("_rem = _dnr.d_would_compute(_r)")
    seg = src[i:i + 400]
    assert ("if _rem > 0 and not _dnr.d_direct_extends(self, _r, _rem) and _dnr.divert(self, _r, _rem):"
            in seg)
