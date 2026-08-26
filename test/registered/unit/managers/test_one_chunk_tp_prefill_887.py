"""#887 THE ONE-CHUNK EXCEPTION under ``phase_flip_purity=strict``.

THE RULE, from the user, verbatim (2026-08-25, during the #857 acceptance
boot): *"also wenn es einfach funktioniert wenn die tp phase theoretisch bis zu
einem chunk selbst prefillen darf und es dann einfacher funktionieren wuerde,
dann ist das auch erlaubt"* -- the TP phase may prefill up to ONE chunk itself.
And immediately after, the qualification that draws the line the code must
carry: *"oder wenn das hicache reinladen als prefill gilt, dann darf es das
natuerlich ueber einen chunk hinaus tun"* -- the cap is on COMPUTED prefill;
a restore is unbounded.

THE CLASS THIS FILE PINS: A BINARY GATE WHERE THE RULE IS A BUDGET. The
permission was yes/no at every site that asks it.
``PhasePurity.prefill_allowed_in_tp()`` takes no token argument at all, and its
own docstring says ``strict`` "forces it to 0 and every prefill flips". The
only alternative on offer, ``prefill_in_tp``, caps nothing at the purity layer
and delegates the quantity to the policy's break-even N -- which
WINDOW_TICKET_874.md computes at **13791** tokens, 3.4x the permission the user
gave. Measured either side of that gap (W29-RESULT.md): 153 prefill batches
executed in the TP layout under ``prefill_in_tp``, 0 under ``strict``. A yes/no
gate cannot express "yes, once", so the truth in between was unreachable -- not
because anybody decided against it, but because no callsite had a number to
hand the gate.

WHAT #870 ALREADY SETTLED, AND IS NOT RE-LITIGATED HERE. The DETECTOR half of
this exists on ``probe/870-detector-modes`` (c5d298149e, 2026-08-26), a leaf
branch that is not an ancestor of the pin. Its discriminator is adopted
verbatim because it is measured and correct: the test is strict ``<``, never
``<=``. A batch that REACHES the chunk size was truncated by it, so more
prefill stands behind it -- a large cold prefill being served in TP, which is
the W37-D defect (258 batches at ``#new-token 4096 / #cached-token 0``, exactly
one chunk, AT the cap). A ``<=`` cap would have called all 258 permitted and
disarmed the instrument. The boundary falls on the violation side: a false red
costs a look, a false green costs the instrument.

WHAT #870 DID NOT DO, and is the reason the detector half is rebuilt here
rather than cherry-picked: it excuses the permitted batch by returning ``None``
-- silently. On the #857 boot that turns 165 TP prefill batches into zero
counted events. Not a violation is not the same as not worth seeing: 165 in one
boot is a leak, "one chunk" is one. So the permitted case gets its own verdict
and its own counter, and only the VIOLATION count stays clean.

WHAT MUST STILL BE TRUE AFTERWARDS:

* **Default byte-identical.** ``strict``, ``prefill_in_tp``, ``threshold:<n>``
  and ``off`` answer exactly as before. The exception exists only when an
  operator writes a budget.
* **COMPUTE vs RESTORE stays separated, structurally.** At the gate the
  seam-transport exemption is checked ABOVE the budget, so a restore never
  spends a chunk. At the detector the cap is measured on ``new_tokens``, which
  is computed tokens only -- restored tokens arrive as ``cached_tokens`` and
  are not in that quantity at all. Unbounded restore is a property of WHICH
  NUMBER is capped, not of an exemption clause that could rot.
"""

from __future__ import annotations

import pytest
from sglang.srt.managers import layout_conformance
from sglang.srt.managers.phase_policy import PHASE_PP, PHASE_TP, TP_TO_PP
from sglang.srt.managers.phase_purity import (
    MODE_OFF,
    MODE_PREFILL_IN_TP,
    MODE_STRICT,
    MODE_THRESHOLD,
    PhasePurity,
    PhasePurityError,
    parse_purity,
    prefill_blocked_here,
    tp_compute_budget_remaining,
    tp_compute_chunks_spent,
)

CHUNK = 4096


class _Runtime:
    def __init__(self, epoch: int = 0):
        self.epoch = epoch


class _Sched:
    """The fields the gates read, and nothing else."""

    def __init__(self, phase, purity, enabled=True, epoch=0, pending=170):
        self.server_args = type(
            "A",
            (),
            {
                "enable_phase_flip": enabled,
                "phase_flip_purity": None,
                "chunked_prefill_size": CHUNK,
            },
        )()
        self.phase_flip_active_stack = phase
        self._phase_purity = purity
        self.phase_flip_runtime = _Runtime(epoch)
        self.waiting_queue = []
        #: The scheduler's own accessor, which the gate consults so that it and
        #: the #838 detector permit exactly the same batches. Default is a
        #: sub-chunk prefill: the case the user's permission is FOR.
        self._pending = pending

    def _pending_prefill_tokens(self, inflight=None) -> int:
        return self._pending


# -- parsing: the budget is a NUMBER, written where the mode is written ----


def test_strict_with_a_chunk_budget_parses():
    p = parse_purity("strict:1")
    assert p.mode == MODE_STRICT
    assert p.tp_compute_chunk_budget == 1
    # It is still STRICT. The valve is a bounded exception to the rule, not a
    # different rule -- every guard keyed on `.strict` (the #858b TP-exit
    # refusal, the spill machinery's decode guarantee) must go on seeing it.
    assert p.strict
    assert p.enforced
    assert not p.decode_allowed_in_pp(0)
    assert not p.decode_allowed_in_pp(4)


def test_strict_zero_is_exactly_strict():
    """One representation for one behaviour, as `threshold:0` already is."""
    assert parse_purity("strict:0") == parse_purity("strict")
    assert parse_purity("strict:0").tp_compute_chunk_budget == 0


@pytest.mark.parametrize("raw", ["strict:", "strict:x", "strict:-1", "strict:1.5"])
def test_a_malformed_budget_is_loud(raw):
    """A budget that silently read as 0 would look like an exception the
    operator set and the instance never honours; one that silently read as
    unbounded would break the user's law outright."""
    with pytest.raises(PhasePurityError):
        parse_purity(raw)


def test_the_other_modes_are_untouched():
    """DEFAULT BYTE-IDENTICAL. The budget field exists on every mode and is 0
    on all of them but a written `strict:<n>`."""
    for raw in (None, "", "prefill_in_tp", "off", "threshold:3", "strict"):
        assert parse_purity(raw).tp_compute_chunk_budget == 0
    assert parse_purity(None).mode == MODE_PREFILL_IN_TP
    assert parse_purity("off").mode == MODE_OFF
    assert parse_purity("threshold:3").mode == MODE_THRESHOLD


def test_the_describe_string_names_the_budget():
    """The boot log line quotes `describe()`. A budget that did not appear
    there would be an exception nobody could see was set."""
    assert PhasePurity(mode=MODE_STRICT).describe() == "strict"
    assert PhasePurity(mode=MODE_STRICT, tp_compute_chunk_budget=1).describe() == (
        "strict:1"
    )


# -- the gate: yes, ONCE ---------------------------------------------------


def test_strict_still_forbids_every_prefill_in_tp():
    """The unbudgeted default, unchanged: not one token."""
    sched = _Sched(PHASE_TP, PhasePurity(mode=MODE_STRICT))
    for _ in range(5):
        assert prefill_blocked_here(sched) is True
    assert tp_compute_chunks_spent(sched) == 0


def test_one_chunk_is_admitted_and_then_the_rule_returns():
    """THE HEADLINE. Exactly one chunk, then strict again -- in ONE TP phase.

    This is the assertion the two shipped alternatives both miss, in opposite
    directions: `strict` never reaches the first False, `prefill_in_tp` never
    reaches the following True.
    """
    purity = PhasePurity(mode=MODE_STRICT, tp_compute_chunk_budget=1)
    sched = _Sched(PHASE_TP, purity)
    assert tp_compute_budget_remaining(sched) == 1
    assert prefill_blocked_here(sched) is False  # the allowed chunk
    assert tp_compute_chunks_spent(sched) == 1
    assert tp_compute_budget_remaining(sched) == 0
    for _ in range(4):
        assert prefill_blocked_here(sched) is True  # and the rule is back


def test_the_budget_is_per_tp_phase_not_per_process():
    """ "Die TP-Phase darf bis zu EINEM Chunk selbst prefillen" -- per phase.

    Keyed on the flip epoch rather than cleared by a reset hook: an epoch that
    advances IS the cutover, and a ledger nobody has to remember to reset
    cannot drift from the event it is scoped to.
    """
    purity = PhasePurity(mode=MODE_STRICT, tp_compute_chunk_budget=1)
    sched = _Sched(PHASE_TP, purity, epoch=7)
    assert prefill_blocked_here(sched) is False
    assert prefill_blocked_here(sched) is True
    sched.phase_flip_runtime.epoch = 8  # a cutover happened
    assert tp_compute_chunks_spent(sched) == 0
    assert prefill_blocked_here(sched) is False
    assert prefill_blocked_here(sched) is True


def test_a_larger_budget_admits_exactly_that_many():
    purity = PhasePurity(mode=MODE_STRICT, tp_compute_chunk_budget=3)
    sched = _Sched(PHASE_TP, purity)
    assert [prefill_blocked_here(sched) for _ in range(5)] == [
        False,
        False,
        False,
        True,
        True,
    ]


def test_the_budget_binds_only_in_the_tp_layout():
    """Prefill in PP is where prefill belongs; it must never spend the valve."""
    purity = PhasePurity(mode=MODE_STRICT, tp_compute_chunk_budget=1)
    pp = _Sched(PHASE_PP, purity)
    for _ in range(5):
        assert prefill_blocked_here(pp) is False
    assert tp_compute_chunks_spent(pp) == 0


def test_a_restore_never_spends_the_compute_budget():
    """THE USER'S OWN SEPARATION, and it is the half that is easy to get wrong.

    *"wenn das hicache reinladen als prefill gilt, dann darf es das natuerlich
    ueber einen chunk hinaus tun"*. The seam-transport exemption is checked
    ABOVE the budget in `prefill_blocked_here`, so a verified restore passes
    unbounded and leaves the one computed chunk still available.
    """
    from sglang.srt.managers.phase_purity import SEAM_READMIT_ATTR

    purity = PhasePurity(mode=MODE_STRICT, tp_compute_chunk_budget=1)
    sched = _Sched(PHASE_TP, purity)
    req = type("R", (), {})()
    setattr(req, SEAM_READMIT_ATTR, 3)
    req.cache_protected_len = 512  # restore evidence: it WAS computed in PP
    sched.waiting_queue = [req]
    for _ in range(6):
        assert prefill_blocked_here(sched) is False
    assert tp_compute_chunks_spent(sched) == 0, "a restore is not computed work"
    # and the computed chunk is still there once the transport is gone
    sched.waiting_queue = []
    assert prefill_blocked_here(sched) is False
    assert prefill_blocked_here(sched) is True


def test_a_probe_reads_the_budget_and_never_spends_it():
    """`_purity_allows("prefill_in_tp")` asks a HYPOTHETICAL. A probe that
    consumed the valve would empty it without a single batch being built --
    the W33 divergence class, in the currency of the new budget."""
    purity = PhasePurity(mode=MODE_STRICT, tp_compute_chunk_budget=1)
    sched = _Sched(PHASE_TP, purity)
    for _ in range(4):
        assert purity.prefill_allowed_in_tp_now(tp_compute_chunks_spent(sched)) is True
    assert tp_compute_chunks_spent(sched) == 0
    assert prefill_blocked_here(sched) is False
    assert purity.prefill_allowed_in_tp_now(tp_compute_chunks_spent(sched)) is False


def test_the_mode_question_and_the_round_question_stay_separate():
    """`prefill_allowed_in_tp()` is asked by BOOT-TIME sizing and by the policy
    threshold collapse (scheduler.py:672, model_runner_kv_cache_mixin.py:7066).
    Under a budgeted strict those must still read "no" -- the flip is still
    demanded for the pending prefill, and the valve is not a mode change."""
    p = PhasePurity(mode=MODE_STRICT, tp_compute_chunk_budget=1)
    assert p.prefill_allowed_in_tp() is False
    assert p.prefill_allowed_in_tp_now(0) is True


def test_the_gate_refuses_when_MORE_than_a_chunk_is_pending():
    """THE GATE AND THE DETECTOR MUST PERMIT THE SAME BATCHES.

    #870's discriminator calls a batch that REACHES the chunk cap a violation --
    it was truncated by the cap, so more prefill stands behind it and what is
    running in TP is a large cold prefill. A gate that granted the chunk anyway
    would build exactly such a batch and the instance would alarm on the
    exception it was configured to take, once per TP phase. It is also the case
    the user did NOT ask for: with a cutover coming regardless, computing 4096
    tokens at TP's 1681 tok/s rather than PP's 7245 spends ~1.9 s of the slow
    layout on work the flip was about to do properly.
    """
    purity = PhasePurity(mode=MODE_STRICT, tp_compute_chunk_budget=1)
    for pending in (CHUNK, CHUNK + 1, 100_000):
        sched = _Sched(PHASE_TP, purity, pending=pending)
        assert prefill_blocked_here(sched) is True, pending
        assert tp_compute_chunks_spent(sched) == 0, "a refusal spends nothing"
    # and the case it IS for: the whole remaining prefill fits, no flip needed
    sched = _Sched(PHASE_TP, purity, pending=CHUNK - 1)
    assert prefill_blocked_here(sched) is False


def test_an_unreadable_pending_count_refuses_rather_than_permits():
    """Unknown is refusal, the same direction as an unresolved chunk size at
    the detector. A stand-in without the accessor, a chunk size of 0, and a
    zero pending count all leave the strict rule standing."""
    purity = PhasePurity(mode=MODE_STRICT, tp_compute_chunk_budget=1)

    no_accessor = _Sched(PHASE_TP, purity)
    del no_accessor.__class__._pending_prefill_tokens  # noqa: B018 - restored below
    try:
        assert prefill_blocked_here(no_accessor) is True
    finally:
        _Sched._pending_prefill_tokens = lambda self, inflight=None: self._pending

    no_chunk = _Sched(PHASE_TP, purity)
    no_chunk.server_args.chunked_prefill_size = 0
    assert prefill_blocked_here(no_chunk) is True

    assert prefill_blocked_here(_Sched(PHASE_TP, purity, pending=0)) is True


# -- #858b: the deadlock term must see the valve ---------------------------


def test_the_858b_runnability_term_sees_a_remaining_budget():
    """`prefill_runnable_in_current_layout` answers "can a pending prefill make
    progress in the layout we are in NOW". With a chunk still owed it CAN, and
    a hold that ignores that is the #858b wedge with a valve installed and
    unreachable -- the W31 shape, exactly."""
    from sglang.srt.managers.phase_flip_runtime import (
        prefill_runnable_in_current_layout,
    )

    strict = PhasePurity(mode=MODE_STRICT)
    budgeted = PhasePurity(mode=MODE_STRICT, tp_compute_chunk_budget=1)
    assert prefill_runnable_in_current_layout(TP_TO_PP, strict) is False
    assert (
        prefill_runnable_in_current_layout(TP_TO_PP, budgeted, budget_remaining=1)
        is True
    )
    # spent: back to the strict answer
    assert (
        prefill_runnable_in_current_layout(TP_TO_PP, budgeted, budget_remaining=0)
        is False
    )


# -- the #838 detector: DISTINGUISH, do not fall silent --------------------


@pytest.fixture(autouse=True)
def _fresh_counters():
    layout_conformance.reset_for_test()
    yield
    layout_conformance.reset_for_test()


def _verdict(**kw):
    base = {
        "batch_class": "prefill",
        "phase": "tp",
        "strict": True,
        "transport_verified": False,
        "n_reqs": 1,
        "new_tokens": CHUNK,
        "cached_tokens": 0,
        "now": 0.0,
    }
    base.update(kw)
    return layout_conformance.work_layout_verdict(**base)


def _exception_note(**kw):
    base = {
        "batch_class": "prefill",
        "phase": "tp",
        "new_tokens": 170,
        "chunk_tokens": CHUNK,
        "budget_configured": 1,
    }
    base.update(kw)
    return layout_conformance.tp_compute_exception_verdict(**base)


def test_the_detector_still_flags_an_unbudgeted_tp_prefill():
    """The W37-D shape, unchanged when no chunk size is supplied."""
    detail = _verdict()
    assert detail is not None
    assert "work_in_wrong_layout" in detail


def test_a_batch_AT_the_chunk_size_is_still_a_violation():
    """#870's load-bearing boundary, adopted verbatim. "Up to one chunk"
    invites `<=`; `<=` calls W37-D's 258 batches permitted and disarms the
    instrument. A batch that REACHES the cap was truncated by it, so more
    prefill stands behind it."""
    assert _exception_note(new_tokens=CHUNK) is None
    detail = _verdict(new_tokens=CHUNK, chunk_tokens=CHUNK, budget_configured=1)
    assert detail is not None
    assert "work_in_wrong_layout" in detail
    assert _exception_note(new_tokens=CHUNK + 1) is None


def test_the_permitted_sub_chunk_batch_is_not_a_violation_but_IS_NAMED():
    """DISTINGUISH, NOT SILENCE. #870 excused this by returning None, which
    turned the #857 boot's 165 TP prefill batches into zero counted events.
    Not-a-violation is not the same as not-worth-seeing: 165 in one boot is a
    leak, "one chunk" is one."""
    note = _exception_note(new_tokens=170)
    assert note is not None
    assert layout_conformance.ALLOWED_TP_COMPUTE in note
    assert "budgeted=yes" in note
    # the violation verdict DEFERS to the same authority
    assert _verdict(new_tokens=170, chunk_tokens=CHUNK, budget_configured=1) is None
    layout_conformance.note_tp_compute_exception(note, 0.0)
    assert layout_conformance.counters().tp_compute_exceptions == 1
    assert layout_conformance.counters().conformance_violations == 0
    assert "tpc=1" in layout_conformance.counters().as_field()


def test_a_sub_chunk_batch_with_NO_valve_configured_is_named_as_such():
    """The #857 signature: strict, no budget flag, and TP prefill happening
    anyway (through the seam door). Under the user's law a sub-chunk compute is
    still not a violation -- but the operator must be able to tell it from the
    valve they deliberately opened."""
    note = _exception_note(budget_configured=0)
    assert note is not None
    assert "budgeted=no" in note
    # AND the positive case must NOT read as the negative one. Asserted as a
    # pair because the one-sided version passed in both worlds: the note's own
    # explanatory tail used to restate the token, so `"budgeted=no" in note`
    # was true regardless of the flag. Caught by mutation_proof_887.
    assert "budgeted=no" not in _exception_note(budget_configured=1)
    assert "budgeted=yes" not in note


def test_an_unknown_chunk_size_never_excuses_anything():
    """`chunk_tokens=None` means the caller could not resolve it. The safe
    reading is "no exception", never "any size fits" -- and it is what every
    pre-#887 caller gets, which is what keeps the default byte-identical."""
    assert _exception_note(chunk_tokens=None) is None
    assert _exception_note(chunk_tokens=0) is None
    assert _verdict(new_tokens=170) is not None


def test_a_zero_token_batch_is_not_an_exception():
    """`new_tokens == 0` computed nothing, so there is nothing to excuse; it
    also must not be excused into silence by a `0 < budget` reading."""
    assert _exception_note(new_tokens=0) is None


def test_a_transport_claim_that_restored_is_excused_before_the_budget():
    """Unchanged #861k precedence: a verified restore that actually restored is
    mechanics, and it never consults the compute cap at any magnitude."""
    assert (
        _verdict(
            transport_verified=True,
            cached_tokens=1_000_000,
            new_tokens=1,
            chunk_tokens=CHUNK,
            budget_configured=1,
        )
        is None
    )


def test_a_genuine_RESTORE_is_not_counted_as_a_compute_exception():
    """THE PRECEDENCE MUST LIVE IN THE AUTHORITY, not in the caller's `elif`.

    A verified restore of a few hundred tokens is sub-chunk, so the compute
    exception's own size test says yes -- and the metrics caller asks this
    function directly to decide whether to book a `tpc` event. Without the
    transport clause inside it, every genuine restore would be counted against
    the one counter whose purpose is to measure COMPUTED work in the wrong
    layout. Found by reading the wiring against its own comment, which asserted
    this precedence in prose and did not implement it.
    """
    assert (
        _exception_note(new_tokens=170, transport_verified=True, cached_tokens=900)
        is None
    )
    # ... and a transport claim that restored NOTHING is not excused by it:
    # the batch is judged on the compute cap like any other computed work.
    assert (
        _exception_note(new_tokens=170, transport_verified=True, cached_tokens=0)
        is not None
    )


def test_a_transport_claim_that_recomputes_a_FULL_chunk_is_a_violation():
    """W37-G must not become excusable by borrowing the new budget."""
    detail = _verdict(
        transport_verified=True,
        cached_tokens=0,
        new_tokens=CHUNK,
        chunk_tokens=CHUNK,
        budget_configured=1,
    )
    assert detail is not None
    assert "transport_claimed=True" in detail


def test_the_detector_defaults_are_byte_identical():
    """Every pre-#887 caller passes neither chunk size nor budget. The verdict
    they get must be the one they got before this ticket."""
    assert _verdict() is not None
    assert _verdict(strict=False) is None
    assert _verdict(phase="pp") is None
    assert _verdict(transport_verified=True, cached_tokens=99) is None
    assert _verdict(new_tokens=0) is not None  # claimed nothing, cached nothing


def test_decode_in_pp_is_never_excused_by_the_prefill_budget():
    """The valve is one-directional. Decode in the PP layout is the half of the
    2026-08-09 rule the starvation measurement actually indicts, and no chunk
    budget touches it."""
    assert _exception_note(batch_class="decode", phase="pp", new_tokens=1) is None
    detail = _verdict(
        batch_class="decode",
        phase="pp",
        new_tokens=1,
        chunk_tokens=CHUNK,
        budget_configured=1,
    )
    assert detail is not None
    assert "class=decode" in detail
