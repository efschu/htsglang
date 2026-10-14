"""PW-W (NF h14 ba1d5becd7, D 14:50:53-14:56:52Z, weg2-2-21): the structural
wedge. The group-decided H105 gate tuple refused the HOL head 1151 times with
``price=167776 budget=33536`` and nothing running; PW-R needed the backup wall
and the workers' own view (4704), H106 was not built under the token cut. The
head now goes back to the front (W50) after N consecutive such evaluations."""

import types

from sglang.srt.weg2 import d_wall_head as dw

RID = "weg2-2-21"


def _sched():
    return types.SimpleNamespace()


def _queue():
    return [types.SimpleNamespace(rid=RID), types.SimpleNamespace(rid="weg2-0-3")]


def _pick(s, verdict, running_empty=True, limit=3, head=RID):
    return dw.pick_structural_wedge_head(sched=s, head_rid=head, waiting=_queue(),
                                         verdict=verdict, running_empty=running_empty,
                                         limit=limit)


V = (RID, "NO_TOKEN", 167776, 33536)


def test_the_head_is_handed_back_after_n_identical_refusals():
    s = _sched()
    assert _pick(s, V) is None
    assert _pick(s, V) is None
    got = _pick(s, V)
    assert got is not None and got.rid == RID
    assert _pick(s, V) is None, "the counter starts over after a hand-back"


def test_a_moved_number_an_admit_or_a_running_request_resets():
    s = _sched()
    _pick(s, V), _pick(s, V)
    assert _pick(s, (RID, "NO_TOKEN", 167776, 40000)) is None   # budget moved: count 1
    assert _pick(s, (RID, "NO_TOKEN", 167776, 40000)) is None   # 2
    assert _pick(s, (RID, "ADMIT", 4704, 33536)) is None        # reset
    _pick(s, V), _pick(s, V)
    assert _pick(s, V, running_empty=False) is None             # reset
    assert _pick(s, V) is None and _pick(s, V) is None


def test_a_head_that_fits_its_budget_is_never_handed_back():
    s = _sched()
    for _ in range(10):
        assert _pick(s, (RID, "NO_TOKEN", 1000, 33536)) is None


def test_the_verdict_must_name_the_head():
    s = _sched()
    for _ in range(10):
        assert _pick(s, ("weg2-0-3", "NO_TOKEN", 83239, 33536)) is None


def test_off_and_default(monkeypatch):
    s = _sched()
    for _ in range(5):
        assert _pick(s, V, limit=0) is None
    monkeypatch.delenv("SGLANG_WEG2_D_WEDGE_REROUTE_N", raising=False)
    assert dw.wedge_reroute_n() == 64
    monkeypatch.setenv("SGLANG_WEG2_D_WEDGE_REROUTE_N", "0")
    assert dw.wedge_reroute_n() == 0
