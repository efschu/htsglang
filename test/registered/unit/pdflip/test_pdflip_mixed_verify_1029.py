"""MIXED-VERIFY stage 1 (29.10.): where the MIXED-D gate says yes on a DFLASH target, the running decode batch runs its OWN speculative
round (draft + verify + accept + commit, the production decode path) and the prefill batch follows in the same scheduler pass -- instead
of the decodes becoming 1-token extend rows (PR #36933 form, 1 token per row, draft state dropped).

Design: deskq/bl/mixed_verify_entwurf_1029.md. Nothing here runs a model: the numeric claim "per decode row bit-identical to a separate run"
holds BY CONSTRUCTION (the decode batch is the decode batch, same worker code) and is checked here as a structural property -- the mixed
step never touches the decode batch's rows, never routes them through ``mix_with_running``, and hands the pair to the same two result
processors. What only metal can show (named in the report): the two-forward order on a real stream, ring pinning, event order of
``publish_ready``, the T_d cost.

Mutation notes (each test names the diff that turns it red; the mutations were run by hand before commit, see the report)."""
from __future__ import annotations

import os
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402
from flliper.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from flliper.srt.pdflip import mixed_verify as mv  # noqa: E402

ARMED = dict(mixed_d_short=True, is_dflash=True, pp_size=1, dp_attention=False, two_batch_overlap=False, pdmux=False)


# ------------------------------------------------------------------------------------------------------------ arm
def test_default_is_on_and_the_notaus_is_the_switch(monkeypatch):
    """New feature -> default ON (user order 14.10.); ``=0`` must reach the old tail mix. Red if the default flips or the
    switch stops being read."""
    assert envs.FLLIPER_PDFLIP_MIXED_VERIFY.default is True
    assert mv.armed(**ARMED) is True
    monkeypatch.setenv("FLLIPER_PDFLIP_MIXED_VERIFY", "0")
    assert mv.enabled() is False
    assert mv.armed(**ARMED) is False


@pytest.mark.parametrize(
    "term",
    ["mixed_d_short", "is_dflash"],
)
def test_arm_needs_the_gate_and_the_dflash_worker(term):
    """Only the MIXED-D gate decides WHEN, and only the DFLASH decode round was reviewed. Red if either term is dropped
    from ``armed`` (EAGLE/NGRAM would then run a decode round the stage was never read against)."""
    assert mv.armed(**{**ARMED, term: False}) is False


@pytest.mark.parametrize(
    "kw",
    [dict(pp_size=2), dict(pp_size=3), dict(dp_attention=True), dict(two_batch_overlap=True), dict(pdmux=True)],
)
def test_arm_refuses_the_loops_that_build_or_split_batches_elsewhere(kw):
    """PP loops build their batches in ``scheduler_pp_mixin`` (they never pass our ``run_batch`` pairing), DP attention syncs the batch
    before the forward, TBO splits it, pdmux splits prefill. Red if one of these terms is dropped: the pair would be lost or
    half-synchronised there."""
    assert mv.armed(**{**ARMED, **kw}) is False


def test_the_ring_is_sized_for_two_forwards_per_pass():
    """``record_batch_in_overlap`` pins each forward's tensors for the one-pass result offset: with two forwards per pass the ring
    needs twice the slots (the DECOUPLE S4b comment at ``init_overlap`` states the one-forward assumption). Red if the armed size
    drops below 2 x default."""
    assert mv.RING_SLOTS_ARMED >= 2 * mv.RING_SLOTS_DEFAULT


# ---------------------------------------------------------------------------------------------------- eligibility
def test_eligibility_terms_and_their_order():
    """A grammar row keeps the old path (overlap + spec + grammar syncs the loop per decode batch); a running batch without a
    DFlashDraftInputV2 would raise in the worker's decode branch. Red if a term is dropped or the census names move."""
    ok = dict(n_running=3, has_grammar=False, spec_state_ok=True)
    assert mv.refusal_reason(**ok) == ""
    assert mv.refusal_reason(**{**ok, "n_running": 1}) == ""              # a single decode request is a full decode batch
    assert mv.refusal_reason(**{**ok, "n_running": 0}) == "no_running"
    assert mv.refusal_reason(**{**ok, "has_grammar": True}) == "grammar"
    assert mv.refusal_reason(**{**ok, "spec_state_ok": False}) == "no_draft_state"
    assert mv.refusal_reason(**{**ok, "pp_size": 3}) == "pp_layout"        # a flip process that is in its PP layout NOW
    assert mv.refusal_reason(**{**ok, "pp_size": 1}) == ""
    assert mv.refusal_reason(n_running=0, has_grammar=True, spec_state_ok=False) == "no_running"   # first matching term


def test_decide_needs_gate_and_every_rank_and_the_process_arm():
    """The pass takes the decode-round form only when the gate agreed AND the group agreed on eligibility AND this process is armed.
    Red if ``decide`` ignores one of the three (a rank-local choice would split the group's batch shape, R11)."""
    assert mv.decide(1, 1, True) is True
    assert mv.decide(0, 1, True) is False
    assert mv.decide(1, 0, True) is False
    assert mv.decide(1, 1, False) is False


# ------------------------------------------------------------------------------------------ the scheduler mix block
class _Batch:
    """Stand-in for the two batches of the mix block; records exactly the calls the block may make."""

    def __init__(self, n, name):
        self.reqs = [types.SimpleNamespace(output_ids=[7]) for _ in range(n)]
        self.name = name
        self.calls = []
        self.batch_is_full = False
        self.req_pool_indices = "rpi-" + name
        self.spec_info = "draft-state-" + name   # the DFlashDraftInputV2 of the round; mix_with_running would null it
        self.mixed_verify_decode = None
        self.decoding_reqs = "unset"
        self.extend_num_tokens = 100

    def filter_batch(self):
        self.calls.append("filter_batch")

    def is_empty(self):
        return not self.reqs

    def batch_size(self):
        return len(self.reqs)

    def prepare_for_decode(self):
        self.calls.append("prepare_for_decode")

    def mix_with_running(self, other):
        self.calls.append(("mix_with_running", other.name))


def _sched(mixed_verify_pass, cross=False, overlap=True):
    from flliper.srt.managers.scheduler import Scheduler

    s = types.SimpleNamespace(
        _mixed_verify_pass=mixed_verify_pass,
        _cross_schedule_mode=cross,
        enable_overlap=overlap,
        spec_algorithm=types.SimpleNamespace(is_none=lambda: False),
        draft_worker=mock.Mock(),
        future_map=mock.Mock(),
        device="cpu",
    )
    return s, Scheduler._fold_running_into_prefill


def test_mixed_verify_pass_keeps_the_decode_batch_a_decode_batch():
    """THE CORE: the decode rows are not folded (no ``mix_with_running``: no 1-token tails, the draft state stays), the decode batch is
    prepared for its round exactly once like the tail block prepares it, it is attached to the prefill batch for ``run_batch``, and it
    stays the pass's running batch. Red if the mixed-verify pass goes through ``mix_with_running`` again or loses the attachment."""
    s, fold = _sched(True)
    new, run = _Batch(2, "new"), _Batch(3, "run")
    out = fold(s, new, run)
    assert out is run                                         # the decode batch stays the running batch of the pass
    assert run.spec_info == "draft-state-run" and run.req_pool_indices == "rpi-run" and len(run.reqs) == 3   # rows + draft state untouched
    assert new.mixed_verify_decode is run                     # ... and rides the prefill batch into run_batch
    assert new.decoding_reqs is None                          # no tail rows: the result processor never takes the tail branch
    assert run.calls == ["filter_batch", "prepare_for_decode"]
    assert not [c for c in new.calls if isinstance(c, tuple)]  # mix_with_running never called


def test_tail_pass_is_the_old_block_byte_for_byte():
    """Switch off / gate-yes-but-ineligible: the old block -- filter, prepare, ``mix_with_running``, ``decoding_reqs`` = the running
    reqs, the running batch emptied, the non-overlap relay stash. Red if the fallback drifts from the 1-token-tail behaviour."""
    s, fold = _sched(False, overlap=False)
    new, run = _Batch(2, "new"), _Batch(3, "run")
    out = fold(s, new, run)
    assert run.calls == ["filter_batch", "prepare_for_decode"]
    assert new.calls == [("mix_with_running", "run")]
    assert new.decoding_reqs is run.reqs
    assert new.mixed_verify_decode is None
    assert isinstance(out, ScheduleBatch) and out.is_empty()
    s.future_map.stash_bonus_tokens.assert_called_once()      # non-overlap spec: the tails' pending tokens


def test_overlap_tail_pass_does_not_stash_the_relay():
    s, fold = _sched(False, overlap=True)
    fold(s, _Batch(1, "new"), _Batch(2, "run"))
    s.future_map.stash_bonus_tokens.assert_not_called()


@pytest.mark.parametrize("verify_pass", [True, False])
def test_every_running_decode_finished_since_the_gate_looked(verify_pass):
    """The gate reads the unfiltered running batch; ``filter_batch`` may empty it. Both forms then fold nothing, prepare nothing and
    return an EMPTY batch (the tail block's old behaviour) -- a mixed-verify pass must not attach an empty decode batch (it would launch
    a decode round of zero rows). Red if the empty case attaches or prepares."""
    s, fold = _sched(verify_pass)
    new, run = _Batch(2, "new"), _Batch(1, "run")
    run.filter_batch = lambda: run.reqs.clear()
    out = fold(s, new, run)
    assert out.is_empty() and out is not run
    assert new.mixed_verify_decode is None and new.decoding_reqs == "unset"
    assert new.calls == []


def test_cross_schedule_rung_is_switched_before_prepare_in_both_forms():
    """T156: the rung is decided BEFORE ``prepare_for_decode`` (the prep dispatches on it). Red if the order flips."""
    for verify_pass in (True, False):
        s, fold = _sched(verify_pass, cross=True)
        order = []
        s.draft_worker.maybe_switch_rung = lambda b: order.append("switch")
        run = _Batch(2, "run")
        run.prepare_for_decode = lambda: order.append("prepare")
        fold(s, _Batch(1, "new"), run)
        assert order == ["switch", "prepare"]


# ------------------------------------------------------------------------------------------ the gate's collective
def _agree_sched(mixed_verify, flags_reply, spec_ok=True, grammar=False):
    from flliper.srt.managers.scheduler import Scheduler

    reqs = [types.SimpleNamespace(weg2_front_halfx=3000)]
    new = types.SimpleNamespace(reqs=reqs, extend_num_tokens=100)
    run = types.SimpleNamespace(reqs=[object(), object()], has_grammar=grammar, spec_info=object())
    s = types.SimpleNamespace(
        _mixed_verify=mixed_verify,
        _mixed_verify_pass=False,
        chunked_req=None,
        server_args=types.SimpleNamespace(tp_prefill_max_tokens=6000),
        ps=types.SimpleNamespace(pp_size=1),
    )
    s._pdflip_group_min_flags = mock.Mock(side_effect=lambda f: flags_reply(f))
    s._mixed_verify_refusal = lambda rb: Scheduler._mixed_verify_refusal(s, rb)
    return s, new, run, Scheduler._mixed_d_agreed


def _patch_spec_state(ok):
    from flliper.srt.speculative import dflash_info_v2

    class _Fake(dflash_info_v2.DFlashDraftInputV2):
        def __init__(self):  # no tensors needed for an isinstance check
            pass

    return _Fake() if ok else object()


def test_the_gate_stays_one_collective_and_gains_the_flag_only_when_armed():
    """R11: the pass's choice between decode-round-plus-prefill and the tail mix is the GROUP's. The second flag rides the gate's single
    all_reduce (one call both ways); off, the call is the old ``[mine]``. Red if a second collective is added, or if the flag is
    evaluated outside the reduce."""
    s, new, run, agreed = _agree_sched(True, lambda f: [1 for _ in f])
    run.spec_info = _patch_spec_state(True)
    assert agreed(s, new, run, True) is True
    assert s._pdflip_group_min_flags.call_count == 1
    assert s._pdflip_group_min_flags.call_args[0][0] == [True, True]
    assert s._mixed_verify_pass is True

    s, new, run, agreed = _agree_sched(False, lambda f: [1 for _ in f])
    assert agreed(s, new, run, True) is True
    assert s._pdflip_group_min_flags.call_args[0][0] == [True]
    assert s._mixed_verify_pass is False


def test_one_rank_ineligible_sends_the_whole_group_to_the_tail_mix():
    """The group MIN: the gate says yes everywhere, but one rank's second flag is 0 (e.g. its running batch lost the draft state) --
    the verdict is still 'mix' (flag 0 yes) while ``_mixed_verify_pass`` is False on EVERY rank, so no rank launches a decode round the
    others fold. Red if the pass flag is taken from the local flag or from the gate alone."""
    s, new, run, agreed = _agree_sched(True, lambda f: [f[0], 0])
    run.spec_info = _patch_spec_state(True)
    assert agreed(s, new, run, True) is True
    assert s._mixed_verify_pass is False


def test_a_grammar_row_is_a_local_no_that_the_group_min_spreads():
    s, new, run, agreed = _agree_sched(True, lambda f: list(f), grammar=True)
    run.spec_info = _patch_spec_state(True)
    assert agreed(s, new, run, True) is True and s._mixed_verify_pass is False
    assert s._pdflip_group_min_flags.call_args[0][0] == [True, False]


def test_no_draft_state_is_a_local_no():
    """A running batch whose ``spec_info`` is not a DFlashDraftInputV2 (cross-algorithm rung, reset) cannot take a DFLASH decode round."""
    s, new, run, agreed = _agree_sched(True, lambda f: list(f))
    run.spec_info = _patch_spec_state(False)
    assert agreed(s, new, run, True) is True and s._mixed_verify_pass is False


def test_gate_no_means_no_decode_round_either():
    """Gate refused (a big prefill): mixed-verify must not run either -- only the gate decides WHEN. Red if ``decide`` forgets the
    gate flag."""
    s, new, run, agreed = _agree_sched(True, lambda f: [0, 1])
    run.spec_info = _patch_spec_state(True)
    assert agreed(s, new, run, True) is False
    assert s._mixed_verify_pass is False


# ------------------------------------------------------------------------------- the run / result pairing (run_batch)
def _run_sched():
    from flliper.srt.managers.scheduler import Scheduler

    s = mock.MagicMock()
    s.run_batch = lambda b, pp=None: Scheduler.run_batch(s, b, pp)
    order = []
    s.seen = []

    def forward(b, pp=None):
        order.append(b.name)
        s.seen.append(b)
        return "result-" + b.name

    s._run_batch_forward = forward
    return s, order, Scheduler


def _sb(name, mode, decode=None):
    b = types.SimpleNamespace(
        name=name, forward_mode=mode, mixed_verify_decode=decode, mixed_verify_pair=None, pdflip_skip_extend=False,
    )
    b.copy = lambda: types.SimpleNamespace(**{**b.__dict__, "name": name + ".copy"})
    return b


def test_run_batch_launches_the_decode_round_first_then_the_prefill_and_pairs_them():
    """ORDER IS THE RANK CONTRACT: decode round, then prefill, on every rank. The decode snapshot copy and its result ride the prefill
    batch as the pair; the attachment is consumed. Red if the order flips, if the pair is not stored, or if the attachment is left
    set (the next ``run_batch`` of the same object would launch the decode round twice)."""
    s, order, _ = _run_sched()
    dec = _sb("dec", ForwardMode.DECODE)
    pre = _sb("pre", ForwardMode.EXTEND, decode=dec)
    result = s.run_batch(pre)
    assert order == ["dec", "pre"]
    assert s.seen[0] is dec and s.seen[1] is pre              # the decode batch goes through the funnel as ITSELF, unwrapped, unmerged
    assert result == "result-pre"
    assert pre.mixed_verify_decode is None
    assert pre.mixed_verify_pair[0].name == "dec.copy" and pre.mixed_verify_pair[1] == "result-dec"


def test_run_batch_without_attachment_is_untouched():
    """Every other batch (plain prefill, plain decode, spill tick) takes exactly the old single launch. Red if ``run_batch`` starts
    storing a pair or launching a second forward for ordinary batches."""
    s, order, _ = _run_sched()
    b = _sb("solo", ForwardMode.DECODE)
    assert s.run_batch(b) == "result-solo"
    assert order == ["solo"] and b.mixed_verify_pair is None


def test_process_batch_result_hands_the_decode_pair_to_the_decode_processor_first():
    """The decode result goes through the SAME entry (decode processor: accept_lens, finish checks, kv release) BEFORE the prefill
    result is processed, and the pair is consumed. Red if the decode result is dropped (its tokens would never reach the user), processed
    after the prefill, or processed twice."""
    from flliper.srt.managers.scheduler import Scheduler

    s = mock.MagicMock()
    s.process_batch_result = lambda b, r: Scheduler.process_batch_result(s, b, r)
    seen = []
    s.batch_result_processor.process_batch_result_decode.side_effect = lambda b, r: seen.append(("decode", b.name, r))
    s.batch_result_processor.process_batch_result_prefill.side_effect = lambda b, r: seen.append(("prefill", b.name, r))
    s.disaggregation_mode = None
    dec = types.SimpleNamespace(name="dec.copy", forward_mode=ForwardMode.DECODE, mixed_verify_pair=None, reqs=[])
    pre = types.SimpleNamespace(
        name="pre", forward_mode=ForwardMode.EXTEND, mixed_verify_pair=(dec, "rdec"), reqs=[],
        is_dllm=lambda: False, decoding_reqs=None,
    )
    Scheduler.process_batch_result(s, pre, "rpre")
    assert seen == [("decode", "dec.copy", "rdec"), ("prefill", "pre", "rpre")]
    assert pre.mixed_verify_pair is None


# --------------------------------------------------------------------------- ScheduleBatch.copy moves the pair
def test_copy_moves_the_pair_to_the_snapshot_and_clears_the_original():
    """The overlap loop processes ``batch.copy()``. If the original kept the pair and later became the running batch (an empty running
    batch is aliased onto the merged prefill batch in ``get_next_batch_to_run``), its next snapshot would hand the OLD decode result to
    the processor again. Red if ``copy`` stops carrying the pair, or leaves it on the original."""
    b = ScheduleBatch(reqs=[], forward_mode=ForwardMode.EXTEND)
    b.mixed_verify_pair = ("dec", "res")
    snap = b.copy()
    assert snap.mixed_verify_pair == ("dec", "res")
    assert b.mixed_verify_pair is None
    assert b.copy().mixed_verify_pair is None


def test_the_attachment_is_not_part_of_the_snapshot():
    """``run_batch`` consumes ``mixed_verify_decode``; a snapshot must never re-launch a decode round."""
    b = ScheduleBatch(reqs=[], forward_mode=ForwardMode.EXTEND)
    b.mixed_verify_decode = ScheduleBatch(reqs=[], forward_mode=ForwardMode.DECODE)
    assert b.copy().mixed_verify_decode is None
