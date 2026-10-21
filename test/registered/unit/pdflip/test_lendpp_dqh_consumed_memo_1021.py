"""LENDPP-1021: no lend return before a head the DQH re-evaluation refuses again.

THE METAL (NF rc27 893d09fa3f live, boot ..._1021_023225.D.log, TP1 02:42:44-02:43:44): 31 pairs of
``D-TRANSIENT-LEND return rows_on 19->1 (before the short extend ...)`` (200-300 ms) and ``lend rows_on 1->19``
(310-480 ms), ``D-DECODE-GAP ... cause=lend extends=0 prefill_tokens=0`` -- not one prefill. The heads pdflip-4-38/
41/43 (resumes, load-back 150912/201216 tokens) were refused at S9 (KV 0.86); DQH held them and evaluated them again
every ``REEVAL_EVERY_PASSES`` (16) held passes, ~2 s.

THE MECHANISM. On the evaluated pass the admission (``_get_new_batch_prefill_raw``) POPS the pass's drain memo
``_pass_prefetch_verdicts``, builds nothing, and DQH re-arms on the fingerprint it took BEFORE (landed prefetches of
the heads included). ``on_batch`` then read the fingerprint without the memo -- landed ``()`` -- saw "changed", took
the queue as takeable and gave the rows back; S93 lent them again after the settle rounds.

What must hold:
(1) the metal window: 31 re-evaluations, the head refused each time -> no return, no re-lend (RED on a773b4ebc4:
    31 + 31);
(2) a re-evaluation that ADMITS (an extend is built) returns the rows before the extend -- the return follows the
    admission's result;
(3) a memo present with a newly landed prefetch is a change: not held (S93 (3) unchanged);
(4) rank unity: three ranks with the same replicated inputs take the same verdict sequence; mutant: the consumed
    memo substituted from a rank-local quantity parts the ranks;
(5) the live witness line ``LENDPP keep`` names the kept hold.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import d_head_hold  # noqa: E402
from flliper.srt.pdflip import d_transient_lend as dtl  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_t_s93_for_lendpp", os.path.join(os.path.dirname(__file__), "test_s93_d_lend_hold_aware_1020.py"))
s93 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s93)

HEADS = ("pdflip-4-38", "pdflip-4-41", "pdflip-4-43")
MEMO = {"pdflip-4-38": True, "pdflip-4-41": True, "pdflip-4-43": False}   # two prefetches landed, still refused
CYCLES = 31


def _decode():
    return s93._decode_batch()


def _extend():
    mode = types.SimpleNamespace(is_decode=lambda: False, is_target_verify=lambda: False, is_idle=lambda: False,
                                 is_extend=lambda: True)
    return types.SimpleNamespace(forward_mode=mode, reqs=[], extend_num_tokens=4096)


def _world(tms):
    r, ctl = s93._lent(tms)
    sched = r.sched
    sched.running_batch = types.SimpleNamespace(reqs=[s93._req("pdflip-4-%d" % i) for i in (10, 11, 12)])
    sched.waiting_queue = [s93._req(rid) for rid in HEADS]
    sched._pdflip_hol = {"rid": HEADS[0]}
    dqh = d_head_hold.DHeadHold(group_d=True, token_cut=lambda: True)
    sched.pdflip_d_head_hold = dqh

    def fp_of(rb, pv):     # the scheduler's _pdflip_d_head_fingerprint, generation constant
        return d_head_hold.fingerprint(hol_state=sched.__dict__.get("_pdflip_hol"), waiting=sched.waiting_queue,
                                       running=rb.reqs if rb is not None else (), prefetch_verdicts=pv,
                                       continuation=sched.chunked_req is not None, generation=7)
    sched._pdflip_d_head_fingerprint = fp_of
    return sched, ctl, dqh


def _iteration(sched, dqh, admits=False):
    """One pass of get_next_batch_to_run as the scheduler orders it: round_start (top), the budget site's drain
    (memo set), DQH in the elif chain, the admission (pops the memo, note_pass), on_batch (end of the funnel)."""
    out = [dtl.round_start(sched)]
    sched._pass_prefetch_verdicts = dict(MEMO)
    if dqh.should_hold(sched._pdflip_d_head_fingerprint(sched.running_batch,
                                                       sched.__dict__.get("_pass_prefetch_verdicts"))):
        batch = _decode()
    else:
        verdicts = sched.__dict__.get("_pass_prefetch_verdicts")
        sched.__dict__.pop("_pass_prefetch_verdicts", None)          # _get_new_batch_prefill_raw consumes it
        batch = _extend() if admits else None
        dqh.note_pass(fp=sched._pdflip_d_head_fingerprint(sched.running_batch, verdicts),
                      built_nothing=batch is None)
        if batch is None:
            batch = _decode()                                          # the running decodes go on
    out.append(dtl.on_batch(sched, batch))
    return out


def _run_window(sched, dqh, n_iter):
    return [v for _ in range(n_iter) for v in _iteration(sched, dqh)]


def _arm(sched, dqh):
    """The window starts held: one evaluated pass refused the heads and armed DQH."""
    sched._pass_prefetch_verdicts = dict(MEMO)
    dqh.should_hold(None)
    verdicts = sched.__dict__.pop("_pass_prefetch_verdicts")
    dqh.note_pass(fp=sched._pdflip_d_head_fingerprint(sched.running_batch, verdicts), built_nothing=True)


ITERS = CYCLES * (d_head_hold.REEVAL_EVERY_PASSES + 1)


# (1) ---------------------------------------------------------------------------------------------------------
def test_metal_window_31_reevaluations_no_return_no_relend():
    tms = s93.h.CoreTms()
    with s93._Ctx(tms):
        sched, ctl, dqh = _world(tms)
        _arm(sched, dqh)
        got = _run_window(sched, dqh, ITERS)
        assert got.count("return") == 0 and got.count("lend") == 0, (got.count("return"), got.count("lend"))
        assert ctl.lent_from == 32 and ctl.rows_on == 40


def test_mutant_the_old_read_reproduces_the_pingpong(monkeypatch):
    """RED shape (a773b4ebc4): the consumed memo read as 'no prefetch landed' -- one return and one re-lend per
    re-evaluation."""
    monkeypatch.setattr(d_head_hold, "same_but_landed", lambda fp, armed: False)
    tms = s93.h.CoreTms()
    with s93._Ctx(tms):
        sched, ctl, dqh = _world(tms)
        _arm(sched, dqh)
        got = _run_window(sched, dqh, ITERS)
        assert got.count("return") >= CYCLES - 1 and got.count("lend") >= CYCLES - 1, \
            (got.count("return"), got.count("lend"))


# (2) ---------------------------------------------------------------------------------------------------------
def test_a_reevaluation_that_admits_returns_before_the_extend():
    tms = s93.h.CoreTms()
    with s93._Ctx(tms):
        sched, ctl, dqh = _world(tms)
        _arm(sched, dqh)
        for _ in range(d_head_hold.REEVAL_EVERY_PASSES):
            assert _iteration(sched, dqh) == [None, None]             # held: the rows stay lent
        got = _iteration(sched, dqh, admits=True)                     # the re-evaluation takes the head
        assert got[1] == "return" and ctl.lent_from is None, got


# (3) ---------------------------------------------------------------------------------------------------------
def test_a_present_memo_with_a_new_landing_is_a_change():
    tms = s93.h.CoreTms()
    with s93._Ctx(tms):
        sched, ctl, dqh = _world(tms)
        _arm(sched, dqh)
        sched._pass_prefetch_verdicts = dict(MEMO, **{"pdflip-4-43": True})
        assert dtl._dqh_holds_queue(sched) is False
        sched.__dict__.pop("_pass_prefetch_verdicts")
        assert dtl._dqh_holds_queue(sched) is True                    # consumed: the armed landing is the latest
        sched.waiting_queue = sched.waiting_queue[:2]                 # another term changed: no hold
        assert dtl._dqh_holds_queue(sched) is False


def test_the_landed_term_index_is_the_fingerprints():
    fp = d_head_hold.fingerprint(hol_state={"rid": "a"}, waiting=[s93._req("a")], running=[],
                                 prefetch_verdicts={"a": True}, continuation=False, generation=3)
    assert fp[d_head_hold.LANDED_TERM] == ("a",)


# (4) ---------------------------------------------------------------------------------------------------------
def _three_ranks(n_iter):
    seqs = []
    for rank in range(3):
        tms = s93.h.CoreTms()
        with s93._Ctx(tms, free=(10 - rank) * s93.STEP_BYTES):     # rank-local card: other amounts, same verdict
            sched, ctl, dqh = _world(tms)
            sched._rank_local_prefetch_done = set(HEADS[:2]) if rank == 0 else set()   # a rank-local fact
            _arm(sched, dqh)
            seqs.append(_run_window(sched, dqh, n_iter))
    return seqs


def test_three_ranks_take_the_same_verdict_sequence():
    seqs = _three_ranks(3 * (d_head_hold.REEVAL_EVERY_PASSES + 1))
    assert seqs[0] == seqs[1] == seqs[2]


def test_mutant_a_rank_local_substitute_parts_the_ranks(monkeypatch):
    """The danger direction: the consumed memo filled from a RANK-LOCAL fact (the rank's own finished prefetches)
    instead of the armed, replicated landed term -- the ranks then give the rows back on different passes (on
    the metal: one rank allocates the extend's activation over lent rows while a peer has them back)."""
    real = d_head_hold.same_but_landed
    current = {}

    def rank_local(fp, armed):
        sched = current["s"]
        return real(fp, armed) and set(armed[d_head_hold.LANDED_TERM]) <= getattr(sched, "_rank_local_prefetch_done")

    orig = dtl._dqh_holds_queue

    def wrapped(sched):
        current["s"] = sched
        return orig(sched)
    monkeypatch.setattr(d_head_hold, "same_but_landed", rank_local)
    monkeypatch.setattr(dtl, "_dqh_holds_queue", wrapped)
    seqs = _three_ranks(3 * (d_head_hold.REEVAL_EVERY_PASSES + 1))
    assert not (seqs[0] == seqs[1] == seqs[2]), "the agreement probe must see the rank-local read"


# (5) ---------------------------------------------------------------------------------------------------------
def test_the_live_witness_names_the_kept_hold(caplog):
    tms = s93.h.CoreTms()
    with s93._Ctx(tms):
        sched, ctl, dqh = _world(tms)
        _arm(sched, dqh)
        caplog.set_level(logging.INFO, logger=dtl.__name__)
        _run_window(sched, dqh, d_head_hold.REEVAL_EVERY_PASSES + 2)
        lines = [m for m in caplog.messages if "LENDPP keep" in m]
        assert lines and "head=pdflip-4-38 waiting=3 landed=2" in lines[0], lines
