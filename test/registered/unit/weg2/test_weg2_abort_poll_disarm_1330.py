"""#1330 -- the abort poll must never read a control word the phase released.

BOOT weg2xsn7 (f) FAIL. The cause is one line ABOVE the traceback:

    WEG2-VRAM-CREDIT tag=weights_7 credit=2560 MiB requested=1906 MiB
    [torch_memory_saver.cpp] CUresult error: 2 (out of memory)
                             func=cu_mem_create line=194
    barlink-BAR1 status poll failed
      self._abort_poll_dst.copy_(self._ctl_dev[0:1], non_blocking=True)
    RuntimeError: unknown parameter type
    Fatal Python error: Segmentation fault

A TMS resume for `weights_7` failed on a DEVICE OOM, leaving the VA behind
`_ctl_dev` unmapped. `copy_` raised, the gate logged and CONTINUED, and the
process then died in the DRIVER -- not at the raise. The segfault is the
amplification; the OOM is the fault.

`barlink_abort_gate.py:317` already named this class before xsn7 was the second
specimen: "the first fault in the whole log is this poll ... three sites, one
fault, and two of them innocent -- which cost this shift two wrong roots."

HERMETIC BY CONSTRUCTION -- no CUDA, no device, no allocator. The guard returns
BEFORE `torch.cuda.stream(...)` is touched, so the whole path under test is
reachable with a stand-in whose storage reports a null data pointer. That is
also why the probe is `untyped_storage().data_ptr()` and not a CUDA call: a
probe that can itself fault is no probe.

MUTANTS (both named by the operator, both must make this file red):
  M1  the probe removed          -> `copy_` reaches the dead tensor
      -> test_no_copy_is_attempted_when_the_backing_is_gone
  M2  the disarm latch removed   -> the poll repeats and re-poisons
      -> test_the_disarm_is_latched_and_the_line_appears_exactly_once
"""

import logging
import types

import pytest

from sglang.srt.distributed.device_communicators import barlink_bar1 as b1

LOGGER = b1.__name__


class _DeadStorage:
    """A released TMS mapping: valid metadata, null data pointer."""

    @staticmethod
    def data_ptr() -> int:
        return 0


class _LiveStorage:
    @staticmethod
    def data_ptr() -> int:
        return 0x7F0000000000


class _CtlWord:
    """Stands in for `_ctl_dev` -- a 2-element int32 device tensor."""

    def __init__(self, storage):
        self._storage = storage
        self.slices = 0

    def untyped_storage(self):
        return self._storage

    def __getitem__(self, item):
        # Reaching here at all means the guard let the read through.
        self.slices += 1
        return self


class _Dst:
    """`_abort_poll_dst`. Its `copy_` is the operation that must NOT happen."""

    def __init__(self):
        self.copies = 0

    def copy_(self, src, non_blocking=False):  # noqa: D401
        self.copies += 1
        raise AssertionError(
            "copy_ was attempted against a released control word -- this is "
            "the call that segfaulted boot weg2xsn7"
        )


def _transport(storage, *, active=True, seen=False):
    """A stand-in carrying exactly the surface `poll_status_word` touches."""
    t = types.SimpleNamespace()
    t._abort_poll_active = active
    t._abort_code_seen = seen
    t._ctl_dev = _CtlWord(storage)
    t._abort_poll_dst = _Dst()
    t._abort_poll_stream = None
    t._round_dev = None
    t._round_mirror = None
    for name in ("poll_status_word", "_abort_poll_disarm"):
        setattr(t, name, getattr(b1.BarlinkBar1Transport, name).__get__(t))
    return t


# --------------------------------------------------------------------------
# M1: the probe
# --------------------------------------------------------------------------


def test_no_copy_is_attempted_when_the_backing_is_gone(caplog):
    """M1: the released mapping is caught BEFORE `copy_`.

    `_Dst.copy_` raises if it is ever reached, so this test cannot pass by
    accident -- removing the probe makes it red with the very message the boot
    log carried.
    """
    t = _transport(_DeadStorage())
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        assert t.poll_status_word() is False
    assert t._abort_poll_dst.copies == 0, "no copy_ may be issued"
    assert t._ctl_dev.slices == 0, "the control word may not even be sliced"
    assert "Bar1AbortPollDisarmed" in caplog.text
    assert "data_ptr=0" in caplog.text


def test_the_disarm_turns_the_poll_off_and_keeps_the_last_verdict(caplog):
    """Disarm, not skip -- and the last known verdict survives.

    The abort word only ever goes 0 -> non-zero and this mirror follows it
    once, so a disarmed poll costs the last reading and nothing more; the hot
    path's view cannot go backwards.
    """
    t = _transport(_DeadStorage())
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        assert t.poll_status_word() is False
    assert t._abort_poll_active is False, "the poll must be OFF, not merely skipped"

    # A transport that had already SEEN a trip keeps answering True forever.
    t2 = _transport(_DeadStorage(), seen=True)
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        assert t2.poll_status_word() is True
    assert t2._abort_poll_dst.copies == 0


# --------------------------------------------------------------------------
# M2: the latch
# --------------------------------------------------------------------------


def test_the_disarm_is_latched_and_the_line_appears_exactly_once(caplog):
    """M2: one line per process, and the poll does not repeat.

    "Log and continue" is the amplifier this fix removes: a poll that failed
    once fails every round. Ten rounds must produce ONE line and ZERO copies.
    """
    t = _transport(_DeadStorage())
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        for _ in range(10):
            assert t.poll_status_word() is False
    assert t._abort_poll_dst.copies == 0
    assert caplog.text.count("Bar1AbortPollDisarmed") == 1, (
        "the disarm must latch -- an unlatched one re-logs every round and, "
        "worse, leaves _abort_poll_active True so the copy is retried"
    )
    # The latch is a real field, not an accident of the log level.
    assert getattr(t, "_abort_poll_disarmed_note", False) is True


def test_the_healthy_path_is_untouched_by_the_probe():
    """The probe costs the working case NOTHING but one metadata read.

    A live storage must fall THROUGH to the staging copy -- proven here by the
    copy being attempted (our stand-in raises to say so). If this ever stops
    raising, the guard has started swallowing healthy polls.
    """
    t = _transport(_LiveStorage())
    with pytest.raises(AssertionError, match="copy_ was attempted"):
        t.poll_status_word()
    assert t._abort_poll_active is True, "a healthy poll must stay armed"
    assert getattr(t, "_abort_poll_disarmed_note", False) is False


def test_an_inactive_or_absent_control_word_returns_before_the_probe():
    """The two pre-existing early exits keep their behaviour exactly."""
    t = _transport(_DeadStorage(), active=False)
    assert t.poll_status_word() is False
    assert getattr(t, "_abort_poll_disarmed_note", False) is False, (
        "an already-inactive poll is not a disarm event and must not log one"
    )
    t2 = _transport(_DeadStorage())
    t2._ctl_dev = None
    assert t2.poll_status_word() is False


def test_the_gate_disarms_the_transport_on_any_poll_failure():
    """The belt: every failure shape the pre-check cannot see in advance.

    A source pin, because the gate's loop needs a transport list, a poison
    record and a logger; what must be true is that the `except` arm reaches
    for the transport's disarm instead of only logging and continuing.
    """
    import inspect

    from sglang.srt.distributed.device_communicators import barlink_abort_gate as g

    src = inspect.getsource(g.poll_status_words)
    assert 'logger.exception("barlink-BAR1 status poll failed")' in src
    assert '_abort_poll_disarm' in src, (
        "logging and continuing is the amplifier -- barlink_abort_gate.py:317 "
        "says so in this module's own words"
    )
    # And it must be reached through a callable check, never a bare attribute
    # access on a transport that may not carry it (the #1298 stand-in trap).
    assert "callable(_disarm)" in src
