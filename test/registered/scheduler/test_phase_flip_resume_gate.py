"""#631: the RESUME GATE -- a disarmed rank waits for the group.

THE DEFECT (reproduced 2026-08-09 06:26:34-35Z, specimen
/spinning/evidence-631/pp_proxy_mispair_20260809T0626Z):

    06:26:34 PP0] FLIP ABANDONED
    06:26:34 PP2] FLIP ABANDONED
    06:26:34 PP1] FLIP ABANDONED          <- LAST
    06:26:35 PP1] #631 PP proxy/batch mismatch: received hidden_states
                  with 1 row(s) for a batch of 24 token(s) (bs=1)

An abandon is RANK-LOCAL -- each rank times out on its own clock. The rank
that disarms first resumes launching and sends its proxy hidden states; a
peer still armed is still withholding, so its cur_batch is None and its
proxy recv (guarded by its OWN batch, never by whether the upstream sent)
does not take the message. It strands, and the proxy stream is purely
positional, so every later receive on that rank is off by one -- silently.
THE RANK THAT DISARMS LAST IS THE RANK THAT STRANDS, and PP1 is both.

THE FIX UNDER TEST: keep withholding past the disarm until every rank has
published its own disarm.

WHY IT CANNOT DEADLOCK, pinned below because it is the first question this
feature's corpse table asks of anything group-wide: the gate BLOCKS ON
NOTHING. It is a predicate re-evaluated per pass, so a gated rank keeps
cycling and servicing its channels. There is no rendezvous to be stuck in.
It is also bounded, so a dead peer cannot hold the survivors for ever.

CPU-only.
"""

import pytest

from sglang.srt.managers.phase_flip_presence import PhaseFlipPresence


def _presence(tmp_path, rank, n_ranks=3):
    return PhaseFlipPresence(
        n_ranks=n_ranks, rank=rank, directory=str(tmp_path), instance="boot"
    )


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _runtime(tmp_path, rank=0, n_ranks=3):
    """A PhaseFlipRuntime with only what the gate touches."""
    from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

    rt = PhaseFlipRuntime.__new__(PhaseFlipRuntime)
    rt._presence = _presence(tmp_path, rank, n_ranks)
    rt._epoch = 0
    rt._clock = _Clock()
    return rt


# -- the marker itself --------------------------------------------------------


def test_disarm_markers_are_per_rank_and_visible_to_peers(tmp_path):
    a, b = _presence(tmp_path, 0), _presence(tmp_path, 1)
    assert a.disarmed(0) == set()
    a.declare_disarmed(0)
    b.declare_disarmed(0)
    assert a.disarmed(0) == {0, 1}
    assert not a.all_disarmed(0), "3 ranks configured, only 2 have disarmed"
    _presence(tmp_path, 2).declare_disarmed(0)
    assert a.all_disarmed(0)


def test_markers_are_epoch_scoped(tmp_path):
    """A later flip must not be released by an earlier one's markers."""
    a = _presence(tmp_path, 0)
    a.declare_disarmed(0)
    assert a.disarmed(1) == set()


# -- the gate -----------------------------------------------------------------


def test_can_fail_a_lone_disarmed_rank_keeps_withholding(tmp_path):
    """THE FALSIFIER. This is the 06:26:34Z situation exactly.

    Rank 0 has abandoned; ranks 1 and 2 have not published yet. Rank 0 must
    NOT resume -- if it does it launches, sends a proxy, and strands it in
    a peer that is still armed.
    """
    rt = _runtime(tmp_path, rank=0)
    rt._open_resume_gate()

    reason = rt.resume_withheld()

    assert reason is not None, (
        "a rank resumed while its peers were still armed; that is the "
        "stranded-proxy defect this gate exists to prevent"
    )
    assert "disarm" in reason


def test_the_gate_opens_the_moment_the_group_has_disarmed(tmp_path):
    rt = _runtime(tmp_path, rank=0)
    rt._open_resume_gate()
    assert rt.resume_withheld() is not None

    _presence(tmp_path, 1).declare_disarmed(0)
    _presence(tmp_path, 2).declare_disarmed(0)

    assert rt.resume_withheld() is None, "the group is disarmed; withholding now costs throughput"


def test_the_gate_is_bounded_and_says_so(tmp_path, caplog):
    """A dead peer may not hold a live rank out of service for ever."""
    rt = _runtime(tmp_path, rank=0)
    rt._open_resume_gate()
    assert rt.resume_withheld() is not None

    rt._clock.t += rt.RESUME_GATE_S + 0.001

    with caplog.at_level("ERROR"):
        assert rt.resume_withheld() is None, "the bound did not release the gate"
    assert "RESUME GATE EXPIRED" in caplog.text, "the bound must be LOUD"


def test_the_expiry_is_reported_once_not_per_pass(tmp_path, caplog):
    """This predicate runs thousands of times a second."""
    rt = _runtime(tmp_path, rank=0)
    rt._open_resume_gate()
    rt._clock.t += rt.RESUME_GATE_S + 0.001
    with caplog.at_level("ERROR"):
        for _ in range(50):
            rt.resume_withheld()
    assert caplog.text.count("RESUME GATE EXPIRED") == 1


def test_it_costs_nothing_when_no_flip_has_been_abandoned(tmp_path):
    """Steady state must not stat /dev/shm on every pass."""
    rt = _runtime(tmp_path, rank=0)
    assert rt.resume_withheld() is None


def test_the_gate_never_waits_on_a_peer(tmp_path):
    """THE DESIGN LAW, pinned: no blocking, so no deadlock class.

    A gate implemented as a rendezvous would hang here -- the peers never
    arrive. It must return promptly with a reason instead, every time.
    """
    rt = _runtime(tmp_path, rank=0)
    rt._open_resume_gate()
    for _ in range(200):
        assert rt.resume_withheld() is not None
