"""#662 -- the amortisation gate must price the TRUE remaining prefill.

RED FIRST. The reported defect is that a long prompt arriving through
CHUNKED PREFILL is prefilled entirely in the TP layout, because the gate
sees only the scheduler-visible slice of the prompt rather than the total
remaining prefill across admitted AND queued requests.

Every case below is written against the LIVE serving configuration, not
against the module defaults, because the defect is a claim about what this
deployment does:

    --chunked-prefill-size 512      (so a 90k prompt is 176 rounds of fill)
    --max-running-requests 4
    N = 7004                        (the measured break-even)
    purity  = prefill_in_tp         (prefill_runs_in_tp True)
    SGLANG_PHASE_POLICY_TP_DECODE_FLOOR_S = 10
    SGLANG_PHASE_POLICY_MIN_DWELL_S       = 3

The NIAH context ladder submits ONE long request at a time and waits for
it, so ``running_bs`` is 0 for the whole prefill: there is no other
generation to strand. That is the case the falsifier pins.
"""

from types import SimpleNamespace

from sglang.srt.managers.phase_policy import (
    PHASE_TP,
    REST_DECODE,
    TP_TO_PP,
    PhasePolicyConfig,
    PhasePolicyState,
    decide,
    PhasePolicyInputs,
)

#: The measured break-even on this rig, as booted.
N_LIVE = 7004

#: The NIAH ladder rows that were measured TP-bound ([7/8], [8/8]).
NIAH_PROMPTS = (58_000, 91_000, 94_000, 120_000)

CHUNK = 512


def live_cfg(**kw):
    """The policy exactly as the shipped boot configures it."""
    base = dict(
        enabled=True,
        flip_tokens=N_LIVE,
        min_dwell_s=3.0,
        idle_dwell_s=20.0,
        rest_state=REST_DECODE,
        tp_decode_floor_s=10.0,
        prefill_runs_in_tp=True,
    )
    base.update(kw)
    return PhasePolicyConfig(**base)


def tp_inputs(pending, running=0, now=1000.0):
    return PhasePolicyInputs(
        phase=PHASE_TP,
        pending_prefill_tokens=pending,
        running_bs=running,
        now=now,
    )


class _StubReq:
    def __init__(self, n):
        self.origin_input_ids = [0] * n


class _StubChunked:
    """A prompt mid-chunked-prefill: admitted, partly computed, NOT queued."""

    def __init__(self, total, filled):
        self.origin_input_ids = [0] * total
        self.extend_range = SimpleNamespace(start=0, end=filled)


def _pending(queue=(), chunked=None):
    """Drive the REAL scheduler metric, never a stub of it."""
    from sglang.srt.managers.scheduler import Scheduler

    class S:
        pass

    s = S()
    s.waiting_queue = list(queue)
    s.chunked_req = chunked
    return Scheduler._pending_prefill_tokens.__get__(s, S)()


# -- the falsifier -------------------------------------------------------------


def test_one_long_request_alone_arms_the_prefill_layout_on_arrival():
    """A 90k prompt sitting in the queue, nothing decoding -> flip to PP.

    This is the state one tick BEFORE admission: the request has been
    received and is queued, so the whole prompt is visible.
    """
    pending = _pending(queue=[_StubReq(91_000)])
    assert pending == 91_000, "a queued prompt must be counted in full"

    d = decide(live_cfg(), PhasePolicyState(), tp_inputs(pending, running=0))
    assert d.direction == TP_TO_PP, (
        f"a 91k prompt with nothing decoding must move to the prefill "
        f"layout, got: {d.reason}"
    )


def test_one_long_request_mid_chunk_still_arms_the_prefill_layout():
    """The SAME prompt one chunk later, now hanging off ``chunked_req``.

    This is the state the defect report names: admitted, 512 tokens
    computed, ~90.5k still to do, and nothing else in the scheduler. The
    gate must price the REMAINDER, not the slice being computed this round.
    """
    for total in NIAH_PROMPTS:
        pending = _pending(chunked=_StubChunked(total, CHUNK))
        assert pending == total - CHUNK, (
            f"the remainder of a {total} tok prompt after one {CHUNK} tok "
            f"chunk must be {total - CHUNK}, got {pending}"
        )

        d = decide(live_cfg(), PhasePolicyState(), tp_inputs(pending, running=0))
        assert d.direction == TP_TO_PP, (
            f"a {total} tok prompt with {pending} tok still to prefill and "
            f"nothing decoding must move to the prefill layout, "
            f"got: {d.reason}"
        )


def test_the_whole_ladder_row_is_priced_not_just_the_visible_chunk():
    """The gate must never see only the chunk-sized slice.

    The wrong implementation prices ``chunked_prefill_size`` (512) and
    compares THAT against N=7004, which can never fire. Pin the sign of the
    comparison explicitly so a regression to slice-pricing is caught here
    rather than on a 4-5x-slow benchmark row.
    """
    d_slice = decide(live_cfg(), PhasePolicyState(), tp_inputs(CHUNK, running=0))
    assert d_slice.direction is None, (
        "a 512 tok slice is genuinely below N; if this armed, the test "
        "below would prove nothing"
    )

    pending = _pending(chunked=_StubChunked(91_000, CHUNK))
    d_total = decide(live_cfg(), PhasePolicyState(), tp_inputs(pending, running=0))
    assert d_total.direction == TP_TO_PP
    assert pending > 100 * CHUNK, (
        "the quantity the gate prices must be the prompt remainder, which "
        "is two orders of magnitude above the chunk"
    )


def test_a_queued_prompt_and_a_chunked_one_are_summed():
    """Admitted + queued, together: the gate prices the SUM.

    A single long request under chunked prefill can coexist with further
    queued arrivals, and neither alone need clear N.
    """
    pending = _pending(
        queue=[_StubReq(4_000), _StubReq(4_000)],
        chunked=_StubChunked(10_000, 9_500),
    )
    assert pending == 4_000 + 4_000 + 500

    d = decide(live_cfg(), PhasePolicyState(), tp_inputs(pending, running=0))
    assert d.direction == TP_TO_PP, (
        f"8500 tok of total remaining prefill clears N={N_LIVE} even though "
        f"no single request does, got: {d.reason}"
    )


def test_amortisation_semantics_are_unchanged_below_the_threshold():
    """The fix must not lower the bar: genuinely small backlogs stay in TP."""
    pending = _pending(chunked=_StubChunked(10_000, 9_000))
    assert pending == 1_000

    d = decide(live_cfg(), PhasePolicyState(), tp_inputs(pending, running=0))
    assert d.direction is None, (
        "1000 tok of remaining prefill does not repay a 4.8 s seam and must "
        "keep running in tp"
    )


def test_decode_floor_still_delays_but_does_not_hide_the_backlog():
    """With decode work resident the floor may DELAY, never blind the gate.

    The reason string must name the floor, not an empty queue: a reason
    that says '0 pending' would mean the visibility defect had returned.
    """
    pending = _pending(chunked=_StubChunked(91_000, CHUNK))
    state = PhasePolicyState()
    state.phase_since = 1000.0
    d = decide(live_cfg(), state, tp_inputs(pending, running=1, now=1001.0))
    assert d.direction is None
    assert "decode floor" in d.reason
    assert str(pending) in d.reason, (
        "the held decision must still report the true backlog it is "
        "deferring, so the gate's input is auditable from the log"
    )
