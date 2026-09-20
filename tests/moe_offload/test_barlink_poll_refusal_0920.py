# SPDX-License-Identifier: Apache-2.0
"""#1489: a bad argument at the barlink poll's extension must never tear the
process down.

THE SPECIMEN (boot weg2xsn406, 2026-09-20 16:49:25Z, D ranks TP0 and TP1).
A manual `POST /weg2/flip` arrived while D was awake. The wake's
`resume_memory_occupation` hit

    [core.cpp] WEG2-TMS-RESUME REFUSED tag=kv_cache rc=2 (out of memory)
               ... every allocation of the tag is PAUSED again

which leaves the tag's physical handles unmapped while the VIRTUAL
reservation stays. The watchdog thread's next pass then did

    barlink_abort_gate.py:433  poll_status_words -> if poll():
    barlink_bar1.py:5696       poll_status_word  -> _abort_poll_dst.copy_(...)
    RuntimeError: unknown parameter type
    Fatal Python error: Segmentation fault

and the rank's own diagnostic printed the pointer that makes #1330's
pre-check useless for this class:

    _ctl_dev=(Tensor dev=cuda:0 dtype=torch.int32 ptr=140017225696256 numel=2)

-- NON-ZERO. `data_ptr() == 0` catches a released mapping, not a paused one.
TP1 logged the raise TWICE in one pass, from two transports, because the gate
disarmed only the transport that raised and the `for` loop walked on.

These tests are hermetic: no CUDA, no transport construction. They drive the
gate's loop with stub transports and the refusal helper with stub tensors,
which is the level the defect lives at -- a loop that continues and an
argument that was never examined.
"""

from __future__ import annotations

import pytest

from sglang.srt.distributed.device_communicators import barlink_abort_gate as gate


class _Transport:
    """Minimal stand-in for a BAR1 transport as the gate's walk sees it."""

    def __init__(self, name, raises=None, trips=False):
        self.name = name
        self._raises = raises
        self._trips = trips
        self.polls = 0
        self.disarmed = None

    def poll_status_word(self):
        self.polls += 1
        if self._raises is not None:
            raise self._raises
        return self._trips

    def _abort_poll_disarm(self, why):
        self.disarmed = why


@pytest.fixture(autouse=True)
def _clean_gate(monkeypatch):
    gate.rearm_gate()
    gate.clear_poison_record()
    monkeypatch.setattr(gate, "_poll_fault_dump", lambda *a, **k: None)
    yield
    gate.rearm_gate()
    gate.clear_poison_record()


def _register(monkeypatch, *transports):
    monkeypatch.setattr(gate, "_transports", list(transports))
    monkeypatch.setattr(gate, "registered", lambda: list(transports))
    monkeypatch.setattr(gate, "abort_check_enabled", lambda: True)


# --- the loop that walked on ------------------------------------------------


def test_raise_stops_the_pass_before_the_next_transport(monkeypatch):
    """The measured amplifier: two tracebacks in ONE pass, then a segfault."""
    a = _Transport("a", raises=RuntimeError("unknown parameter type"))
    b = _Transport("b")
    _register(monkeypatch, a, b)

    assert gate.poll_status_words() == 0

    assert a.polls == 1
    assert b.polls == 0, (
        "the second transport was touched in the same poisoned context -- this "
        "is exactly the second traceback TP1 logged before the segfault"
    )


def test_raise_disarms_the_transport_and_the_whole_gate(monkeypatch):
    a = _Transport("a", raises=RuntimeError("unknown parameter type"))
    _register(monkeypatch, a)

    gate.poll_status_words()

    assert a.disarmed is not None, "#1330's per-transport disarm must still fire"
    why = gate.gate_disarmed()
    assert why is not None, "#1489: one unmapped mapping is a fact about the process"
    assert "unknown parameter type" in why
    assert "RuntimeError" in why


def test_gate_stays_disarmed_on_every_later_round(monkeypatch):
    a = _Transport("a", raises=RuntimeError("unknown parameter type"))
    _register(monkeypatch, a)

    gate.poll_status_words()
    assert a.polls == 1
    for _ in range(5):
        assert gate.poll_status_words() == 0
    assert a.polls == 1, "a poll that failed once must not be asked every round"


def test_disarm_gate_records_the_first_reason_only():
    assert gate.disarm_gate("first") is True
    assert gate.disarm_gate("second") is False
    assert gate.gate_disarmed() == "first"


def test_healthy_pass_is_untouched(monkeypatch):
    a = _Transport("a", trips=True)
    b = _Transport("b", trips=False)
    _register(monkeypatch, a, b)

    assert gate.poll_status_words() == 1
    assert (a.polls, b.polls) == (1, 1)
    assert gate.gate_disarmed() is None


def test_poison_path_still_records_the_origin(monkeypatch):
    """#867 keeps its own handling: the poison RECORD is what attributes the
    later crash, and it must not be replaced by the #1489 latch."""
    a = _Transport("a", raises=RuntimeError("an illegal memory access was encountered"))
    b = _Transport("b")
    _register(monkeypatch, a, b)

    gate.poll_status_words()

    rec = gate.poison_record()
    assert rec is not None and "illegal memory access" in rec["error"]
    assert b.polls == 0


def test_pause_polling_suppresses_the_pass(monkeypatch):
    """The TMS pause/resume chokepoint rides on this exclusion."""
    a = _Transport("a", trips=True)
    _register(monkeypatch, a)
    with gate.pause_polling():
        assert gate.poll_status_words() == 0
        assert a.polls == 0
    assert gate.poll_status_words() == 1


# --- the chokepoint every tag pause/resume goes through ---------------------


def test_tms_adapter_excludes_the_poll_for_the_length_of_the_call(monkeypatch):
    """The window weg2xsn406 fell into: a wake's TMS resume and the 10 ms
    watchdog poll in the same microsecond."""
    from sglang.srt.utils import torch_memory_saver_adapter as tms

    a = _Transport("a", trips=True)
    _register(monkeypatch, a)

    seen = []
    with tms._abort_poll_excluded():
        seen.append(gate.polling_paused())
        seen.append(gate.poll_status_words())
    assert seen == [True, 0]
    assert a.polls == 0
    assert gate.polling_paused() is False
    assert gate.poll_status_words() == 1


def test_tms_adapter_exclusion_never_raises(monkeypatch):
    """A guard that can break bring-up is worse than the gap it closes."""
    import builtins

    from sglang.srt.utils import torch_memory_saver_adapter as tms

    real_import = builtins.__import__

    def _boom(name, *a, **k):
        if name.endswith("device_communicators"):
            raise ImportError("no gate here")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _boom)
    with tms._abort_poll_excluded():
        pass


def test_tms_adapter_exclusion_releases_on_an_exception(monkeypatch):
    from sglang.srt.utils import torch_memory_saver_adapter as tms

    with pytest.raises(ValueError):
        with tms._abort_poll_excluded():
            assert gate.polling_paused() is True
            raise ValueError("resume refused")
    assert gate.polling_paused() is False, (
        "a refused resume must not leave the abort poll muted forever"
    )


# --- the arguments that were never examined ---------------------------------


class _Poll:
    """`BarlinkBar1Transport._abort_poll_arg_refusal` bound to a bare object.

    Taking the unbound function off the class keeps the test free of the
    transport's ctypes/mmap constructor while testing the real code.
    """

    def __init__(self, **kw):
        import torch

        from sglang.srt.distributed.device_communicators.barlink_bar1 import (
            BarlinkBar1Transport,
        )

        self._refusal = BarlinkBar1Transport._abort_poll_arg_refusal
        self._ctl_dev = kw.get("ctl", torch.zeros(2, dtype=torch.int32))
        self._abort_poll_dst = kw.get("dst", torch.zeros(1, dtype=torch.int32))
        self._abort_poll_stream = kw.get("stream", _FakeStream())
        self._round_dev = kw.get("round_dev")
        self._round_mirror = kw.get("round_mirror")

    def __call__(self):
        return self._refusal(self)


class _FakeStream:
    pass


def _cuda_like(t):
    """Make a CPU tensor answer `is_cuda` True. The refusal helper reads
    metadata only, which is the property that lets this test run with
    CUDA_VISIBLE_DEVICES=''."""

    class _Wrap:
        def __init__(self, inner):
            self._inner = inner

        is_cuda = True

        def __getattr__(self, name):
            return getattr(self._inner, name)

    return _Wrap(t)


def test_refusal_names_a_none_control_word():
    p = _Poll(ctl=None)
    assert "not a Tensor" in p() and "_ctl_dev" in p()


def test_refusal_names_a_none_destination():
    import torch

    p = _Poll(ctl=_cuda_like(torch.zeros(2, dtype=torch.int32)), dst=None)
    # A wrapper is not a torch.Tensor, so the ctl check fires first; the point
    # of this pair is that NEITHER shape reaches the extension.
    assert p() is not None


def test_refusal_names_a_host_control_word():
    import torch

    p = _Poll(ctl=torch.zeros(2, dtype=torch.int32))  # is_cuda False
    why = p()
    assert why is not None and "not on a device" in why


def test_refusal_names_an_int_destination():
    p = _Poll(ctl=None, dst=0)
    assert p() is not None


def test_refusal_names_a_dtype_mismatch(monkeypatch):
    import torch

    from sglang.srt.distributed.device_communicators.barlink_bar1 import (
        BarlinkBar1Transport,
    )

    class _Obj:
        pass

    o = _Obj()
    o._ctl_dev = torch.zeros(2, dtype=torch.int32)
    o._abort_poll_dst = torch.zeros(1, dtype=torch.int64)
    o._abort_poll_stream = torch.cuda.Stream
    o._round_dev = None
    o._round_mirror = None
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.cuda, "Stream", object)
    why = BarlinkBar1Transport._abort_poll_arg_refusal(o)
    assert why is not None
    assert "dtype mismatch" in why


def test_refusal_passes_a_sound_set(monkeypatch):
    import torch

    from sglang.srt.distributed.device_communicators.barlink_bar1 import (
        BarlinkBar1Transport,
    )

    class _Obj:
        pass

    o = _Obj()
    o._ctl_dev = torch.zeros(2, dtype=torch.int32)
    o._abort_poll_dst = torch.zeros(1, dtype=torch.int32)
    o._abort_poll_stream = object()
    o._round_dev = None
    o._round_mirror = None
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.cuda, "Stream", object)
    assert BarlinkBar1Transport._abort_poll_arg_refusal(o) is None
