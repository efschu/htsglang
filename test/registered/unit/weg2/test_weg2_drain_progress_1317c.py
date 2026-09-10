"""#1317c -- the drain may not call a DECODING request a wedge.

SPECIMEN, boot weg2sn6e (2026-09-10, D log
`boot_weg2_weg2sn6e_82cad748c2_0910_085435.D.log`): the front refused three
120 s drain windows with `W1 Weg2DrainRefused: D still holds 1 request(s)`
and escalated to `W2 Weg2DrainStuck -> WEG2 STOP`, health 503. That request
was decoding normally the whole time: **686 `Decode batch` lines**, `#full
token` climbing monotonically **8,814 -> 84,384** (~75,600 tokens),
`#running-req: 1`. Every refusal was TRUE as stated and WRONG as used -- the
predicate reads RESIDENCY and is consumed as a WEDGE verdict, so a long
generation and a livelock were indistinguishable to it.

THE LAW THAT DECIDES IT (#1011, standing user order, verbatim: "er decoded
zuende ... und flippt dann zurueck"): a decode runs to the end and is never
cut. A decoding request is therefore never a wedge.

THE GUARD KEEPS ITS TEETH, which is the other half of this file: W1 still
counts a NO-PROGRESS window and three of those still reach W2. The danger
direction of THIS change is the opposite of the last one -- not a false
wedge, but a guard talked out of firing -- so the mutants push that way:
  M1  a livelock that spins forward passes and emits nothing must STILL be
      refused -> test_a_forward_spinning_livelock_is_still_a_refusal
  M2  an unreadable counter must keep the OLD behaviour, never read as
      progress -> test_an_unreadable_counter_falls_back_and_never_waits
  M3  a counter that went backwards must not read as progress
      -> test_a_counter_that_went_backwards_is_not_progress
  M4  a stalled request must still reach W1 x3 -> W2
      -> test_three_no_progress_windows_still_reach_w2
"""


def delta(before, after):
    """Imported INSIDE the call on purpose.

    A module-level import of a symbol the parent does not have turns this
    whole file into ONE collection error, which proves only that the name is
    missing. Per-test red is the stronger red-first signal: on the parent each
    assertion below fails on its own, so the tally says how many distinct
    claims the fix carries.
    """
    from sglang.srt.weg2.front import weg2_drain_progress_delta

    return weg2_drain_progress_delta(before, after)


# The specimen's own counters, from the D log.
SN6E_BEFORE = {"gen_tokens_total": 8814, "prefill_tokens_total": 12000,
               "forward_ct": 100, "running": 1}
SN6E_AFTER = {"gen_tokens_total": 84384, "prefill_tokens_total": 12000,
              "forward_ct": 786, "running": 1}


# --------------------------------------------------------------------------
# the specimen must read as a WAIT
# --------------------------------------------------------------------------

def test_the_sn6e_window_reads_as_progress_not_as_a_refusal():
    d = delta(SN6E_BEFORE, SN6E_AFTER)
    assert d is not None
    assert d["progressed"] is True, (
        "the boot that produced this delta was STOPPED as a wedge"
    )
    # and the numbers are the specimen's, so this test fails if the reader
    # ever stops differencing the counter it claims to difference
    assert d["tokens"] == 84384 - 8814 == 75570
    assert d["forward"] == 686


def test_a_chunk_prefilling_request_is_not_stalled_either():
    """A long prompt still moving through chunked prefill emits no DECODE
    tokens. Counting only decode would call it a wedge -- the same error one
    tier over, so the prefill arm is part of the predicate."""
    d = delta(SN6E_BEFORE, dict(SN6E_BEFORE, prefill_tokens_total=21000))
    assert d["progressed"] is True
    assert d["prefill_tokens"] == 9000
    assert d["tokens"] == 0


# --------------------------------------------------------------------------
# the guard keeps its teeth
# --------------------------------------------------------------------------

def test_a_stalled_window_is_still_a_refusal():
    assert delta(SN6E_BEFORE, dict(SN6E_BEFORE))["progressed"] is False


def test_a_forward_spinning_livelock_is_still_a_refusal():
    """M1, THE DANGER DIRECTION OF THIS CHANGE. `forward_ct` is reported as a
    witness but must never license a wait on its own: a livelock that spins
    forward passes and emits nothing is exactly the wedge the guard exists to
    catch, and crediting it as progress would remove the teeth while looking
    like a safety improvement."""
    d = delta(SN6E_BEFORE, dict(SN6E_BEFORE, forward_ct=9000))
    assert d["forward"] == 8900, "the witness must still be REPORTED"
    assert d["progressed"] is False, (
        "forward passes without tokens are not progress -- this is the wedge"
    )


def test_an_unreadable_counter_falls_back_and_never_waits():
    """M2. None -- never a zero and never a wait. An unreadable counter must
    reach the caller as an ABSENCE so it keeps the old residency behaviour; a
    failed HTTP call read as `progressed` would suppress every W1 for ever,
    and read as `not progressed` would manufacture the very W2 this change
    exists to prevent."""
    assert delta(None, SN6E_AFTER) is None
    assert delta(SN6E_BEFORE, None) is None
    assert delta(None, None) is None
    assert delta("not a dict", SN6E_AFTER) is None


def test_a_counter_that_went_backwards_is_not_progress():
    """M3. Backwards means a restart or a phase rebind, not negative work.
    The honest answer for that window is "cannot say it worked"."""
    d = delta(SN6E_AFTER, SN6E_BEFORE)
    assert d["tokens"] == 0
    assert d["progressed"] is False


def test_three_no_progress_windows_still_reach_w2():
    """M4. The streak arithmetic, as the flip applies it: a working window
    RESETS it, a stalled one increments, and three stalled in a row reach W2.
    Kept as arithmetic beside the predicate so the two cannot drift."""
    def run(windows):
        streak = 0
        for progressed in windows:
            if progressed:
                streak = 0          # WAIT: the streak resets
            else:
                streak += 1
                if streak >= 3:
                    return "W2"
        return f"streak={streak}"

    assert run([False, False, False]) == "W2", "the guard lost its teeth"
    assert run([True] * 20) == "streak=0", "a decode must never reach W2"
    # the sn6e shape: progress every window -> never W2
    assert run([True, True, True, True]) == "streak=0"
    # and a working window in the middle genuinely resets, by design: under
    # #1011 the decode is allowed to take as long as it takes
    assert run([False, False, True, False, False]) == "streak=2"


# --------------------------------------------------------------------------
# the wiring
# --------------------------------------------------------------------------

def test_the_flip_consults_progress_before_counting_w1():
    import inspect

    from sglang.srt.weg2.front import Front

    src = inspect.getsource(Front.flip)
    i_prog = src.index("_drain_progress")
    i_w1 = src.index('self.counters["W1_Weg2DrainRefused"] += 1')
    assert i_prog < i_w1, "W1 is counted before progress is consulted"
    assert "WEG2 DRAIN WAITING" in src
    assert "#1011" in src, "the law that licenses the wait is not named"


def test_the_drain_samples_both_ends_of_the_window():
    import inspect

    from sglang.srt.weg2.front import Front

    src = inspect.getsource(Front.drain)
    assert src.count("_weg2_decode_progress") == 2, (
        "the delta needs a sample at BOTH ends of the window"
    )
    assert "weg2_drain_progress_delta" in src


def test_d_publishes_the_counters_on_the_endpoint_the_front_already_polls():
    """No new endpoint and no new poll: the front already reads
    `internal_states[0]` of `/get_server_info` for the draft terms."""
    import inspect

    from sglang.srt.managers.scheduler import Scheduler

    import ast
    import textwrap

    src = inspect.getsource(Scheduler.get_internal_state)
    assert '"weg2_decode_progress"' in src
    for key in ("gen_tokens_total", "prefill_tokens_total", "forward_ct", "running"):
        assert key in src, f"the progress block lost {key}"
    # gen_tokens_total and NOT num_generated_tokens: the latter is zeroed every
    # logging interval and cannot be differenced over a window.
    #
    # READ THE EXECUTED CODE, NOT THE TEXT -- fourth instance of this class
    # today (the #1318 pins, the extent docstring, the route provenance): the
    # comment IN the function names `num_generated_tokens` precisely in order
    # to reject it, so a source-text grep reports the rejection as a use.
    fn = ast.parse(textwrap.dedent(src)).body[0]
    executed = {
        n.value
        for n in ast.walk(fn)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    } | {
        n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)
    }
    assert "gen_tokens_total" in executed
    assert "num_generated_tokens" not in executed, (
        "the progress block must not read the counter that is zeroed every "
        "logging interval -- it cannot be differenced over a window"
    )


def test_the_window_length_is_unchanged():
    """The ruling kept the 120 s window; only the VERDICT taken at its end
    changed. A test that let the window drift would hide a different fix."""
    import inspect

    from sglang.srt.weg2 import front

    src = inspect.getsource(front.Front.drain)
    assert "self.drain_deadline_s" in src
    assert "drain_deadline_s" in inspect.getsource(front)
