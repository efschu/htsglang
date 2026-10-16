"""K10 PW-F (NF h16p 15.10. 12:21:52-12:27:33Z, D TP0-2): the no-fund wedge.

7 queued, 0 running, every pass the group-D HOL head refused NO_TOKEN by the H105d cut
(``FORM-A-CUT LOAD-BACK ROOM ... evicted=0``) while the peel paid nothing (backup wall,
``ARENA-DROP freed=0`` x3366). PW-R flapped on a rank-local 2 s memo (12:24:02 TP0
payable=516288, 12:23:44 TP1 payable=0), PW-W needed price > budget, H106 a host refusal.
PW-F counts the pass itself -- replicated inputs only -- and answers the current head W50.
Hermetic, no CUDA.
"""
import inspect
import types

from sglang.srt.weg2 import d_wall_head as dw


def _q():
    return [types.SimpleNamespace(rid="weg2-10-51", full_untruncated_fill_ids=list(range(88705)),
                                  prefix_indices=list(range(2944))),
            types.SimpleNamespace(rid="weg2-21-67", full_untruncated_fill_ids=[], prefix_indices=[])]


def _nf(s, head="weg2-10-51", running_empty=True, admitted=0, limit=3):
    return dw.pick_nofund_head(sched=s, head_rid=head, waiting=_q(), running_empty=running_empty,
                               admitted_n=admitted, limit=limit)


def test_pwf_hands_the_head_back_after_n_no_fund_passes_and_starts_over():
    s = types.SimpleNamespace()
    assert _nf(s) is None and _nf(s) is None
    got = _nf(s)
    assert got is not None and got.rid == "weg2-10-51"
    assert dw.device_need_rows(got) == 88705 - 2944
    assert _nf(s) is None


def test_pwf_a_rotating_head_still_counts_and_the_current_head_goes():
    """K10: the H105d head changed (10-51, 33-80, 33-81) -- the wedge is the pass, not one rid."""
    s = types.SimpleNamespace()
    assert _nf(s, head="weg2-21-67") is None
    assert _nf(s, head="weg2-10-51") is None
    assert _nf(s, head="weg2-21-67").rid == "weg2-21-67"


def test_pwf_an_admit_a_run_or_no_refused_head_resets():
    for kw in ({"admitted": 1}, {"running_empty": False}, {"head": None}):
        s = types.SimpleNamespace()
        _nf(s)
        _nf(s)
        assert _nf(s, **kw) is None
        assert _nf(s) is None and _nf(s) is None, kw
        assert _nf(s) is not None
    s = types.SimpleNamespace()
    _nf(s)
    _nf(s)
    dw.reset_nofund(s)
    assert _nf(s) is None


def test_pwf_zero_is_off(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_D_NOFUND_REROUTE_N", "0")
    s = types.SimpleNamespace()
    for _ in range(100):
        assert dw.pick_nofund_head(sched=s, head_rid="weg2-10-51", waiting=_q(), running_empty=True,
                                   admitted_n=0) is None


class _FlappingTree:
    """Each rank's rank-local wall reading, as K10 measured it: TP0 payable=reported, TP1 0."""

    def __init__(self, payable):
        self.payable = payable

    def evictable_size(self):
        return 516288

    def payable_evictable_size(self):
        return self.payable


def test_pwf_every_rank_answers_the_same_head_in_the_same_pass():
    """Mutant 'rank-local counter' dies here: the ranks carry DIFFERENT tree readings (K10 12:24:02
    TP0 payable=516288, TP1 0, TP2 513280) -- the verdict must not read them."""
    ranks = [types.SimpleNamespace(tree_cache=_FlappingTree(p)) for p in (516288, 0, 513280)]
    verdicts = []
    for _pass in range(5):
        verdicts.append([getattr(_nf(r, limit=4), "rid", None) for r in ranks])
    assert all(len(set(v)) == 1 for v in verdicts), verdicts
    assert [v[0] for v in verdicts] == [None, None, None, "weg2-10-51", None]


def test_pwf_wiring_after_pw_w_and_reset_on_every_other_pass():
    from sglang.srt.managers import scheduler as sch

    src = inspect.getsource(sch)
    i_w = src.index("_weg2_d_wall_head.pick_structural_wedge_head(")
    i_f = src.index("_weg2_d_wall_head.pick_nofund_head(")
    i_r = src.index("_weg2_d_wall_head.reset_nofund(self)")
    i_a = src.index("self._weg2_answer_x_refusals(_x_refused, _head_inputs, wall_rows=_wall_rows)")
    assert i_w < i_f < i_r < i_a
    assert "admitted_n=len(adder.can_run_list)" in src
