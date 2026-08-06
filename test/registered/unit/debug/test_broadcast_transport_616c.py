"""Unit tests for the broadcast-transport path in capture_safe_tp_broadcast
and spec_accept_broadcast_src.

Hermetic: no CUDA, no torch tensors, no distributed. Uses plain Python
sentinel objects and call-recording fakes.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import List, Tuple


class FakeChangeState:
    """Context manager that records enter/exit for pynccl_comm.change_state."""

    def __init__(self) -> None:
        self.entered: bool = False
        self.exited: bool = False

    def __enter__(self) -> "FakeChangeState":
        self.entered = True
        return self

    def __exit__(self, *args: object) -> None:
        self.exited = True


class FakePyncclComm:
    """Minimal fake of a pynccl communicator."""

    def __init__(self, available: bool = True) -> None:
        self.available: bool = available
        self.broadcast_calls: List[Tuple[object, int]] = []
        self._change_state_obj: FakeChangeState = FakeChangeState()

    @contextmanager
    def change_state(self, enable: bool = True):
        """Yields a context manager that tracks enter/exit."""
        obj = FakeChangeState()
        with obj:
            yield obj

        # After the context exits, store so the test can inspect it
        self._change_state_obj = obj

    def broadcast(self, tensor: object, src: int = 0) -> None:
        self.broadcast_calls.append((tensor, src))


class FakeTpGroup:
    """Minimal fake of a GroupCoordinator / tp_group."""

    def __init__(self, pynccl_comm: FakePyncclComm | None = None) -> None:
        self.pynccl_comm = pynccl_comm
        self.broadcast_calls: List[Tuple[object, int]] = []

    def broadcast(self, tensor: object, src: int = 0) -> None:
        self.broadcast_calls.append((tensor, src))


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_sentinel(name: str) -> object:
    """Return a distinct plain object acting as a tensor placeholder."""
    return name  # string sentinels are hashable and distinct enough


# ---------------------------------------------------------------------------
# capture_safe_tp_broadcast  -- pynccl available (Test 1)
# ---------------------------------------------------------------------------


def test_pynccl_broadcast_used_when_available() -> None:
    """When pynccl_comm exists and is available, every non-None tensor is
    broadcast via pynccl_comm.broadcast and tp_group.broadcast is NOT called."""
    comm = FakePyncclComm(available=True)
    tp_group = FakeTpGroup(pynccl_comm=comm)

    from sglang.srt.speculative.spec_utils import capture_safe_tp_broadcast

    tensors = (_make_sentinel("t0"), _make_sentinel("t1"), _make_sentinel("t2"))
    capture_safe_tp_broadcast(tp_group, tensors, src=3)

    # pynccl path: 3 non-None tensors
    assert len(comm.broadcast_calls) == 3, (
        f"expected 3 pynccl broadcasts, got {len(comm.broadcast_calls)}"
    )
    for i, (t, src) in enumerate(comm.broadcast_calls):
        assert t == f"t{i}", f"unexpected tensor {t!r}"
        assert src == 3, f"expected src=3, got {src}"

    # tp_group fallback must NOT have been called
    assert len(tp_group.broadcast_calls) == 0, (
        f"tp_group.broadcast called {len(tp_group.broadcast_calls)} times; "
        "expected 0 on the pynccl path"
    )


# ---------------------------------------------------------------------------
# capture_safe_tp_broadcast  -- pynccl absent (Test 2)
# ---------------------------------------------------------------------------


def test_c10d_fallback_when_pynccl_none() -> None:
    """When pynccl_comm is None, tp_group.broadcast is used for every
    non-None tensor."""
    tp_group = FakeTpGroup(pynccl_comm=None)

    from sglang.srt.speculative.spec_utils import capture_safe_tp_broadcast

    tensors = (_make_sentinel("a"), _make_sentinel("b"))
    capture_safe_tp_broadcast(tp_group, tensors, src=7)

    assert len(tp_group.broadcast_calls) == 2
    for i, (t, src) in enumerate(tp_group.broadcast_calls):
        assert t in ("a", "b")
        assert src == 7

    # pynccl must not have been touched
    assert getattr(tp_group, "pynccl_comm", None) is None


# ---------------------------------------------------------------------------
# capture_safe_tp_broadcast  -- pynccl exists but not available (Test 3)
# ---------------------------------------------------------------------------


def test_c10d_fallback_when_pynccl_unavailable() -> None:
    """When pynccl_comm exists but .available is False, the c10d fallback
    (tp_group.broadcast) is used."""
    comm = FakePyncclComm(available=False)
    tp_group = FakeTpGroup(pynccl_comm=comm)

    from sglang.srt.speculative.spec_utils import capture_safe_tp_broadcast

    tensors = (_make_sentinel("x"),)
    capture_safe_tp_broadcast(tp_group, tensors, src=0)

    # c10d fallback
    assert len(tp_group.broadcast_calls) == 1
    assert tp_group.broadcast_calls[0][0] == "x"

    # pynccl.broadcast must NOT have been called
    assert len(comm.broadcast_calls) == 0


# ---------------------------------------------------------------------------
# capture_safe_tp_broadcast  -- None entries skipped (Test 4)
# ---------------------------------------------------------------------------


def test_none_entries_skipped_pynccl_path() -> None:
    """None entries in the tensor tuple are skipped on the pynccl path."""
    comm = FakePyncclComm(available=True)
    tp_group = FakeTpGroup(pynccl_comm=comm)

    from sglang.srt.speculative.spec_utils import capture_safe_tp_broadcast

    # 3 tensors, 2 are None
    tensors = (_make_sentinel("real"), None, None)
    capture_safe_tp_broadcast(tp_group, tensors, src=0)

    assert len(comm.broadcast_calls) == 1
    assert comm.broadcast_calls[0][0] == "real"


def test_none_entries_skipped_c10d_path() -> None:
    """None entries in the tensor tuple are skipped on the c10d fallback path."""
    tp_group = FakeTpGroup(pynccl_comm=None)

    from sglang.srt.speculative.spec_utils import capture_safe_tp_broadcast

    tensors = (None, _make_sentinel("real"), None)
    capture_safe_tp_broadcast(tp_group, tensors, src=0)

    assert len(tp_group.broadcast_calls) == 1
    assert tp_group.broadcast_calls[0][0] == "real"


# ---------------------------------------------------------------------------
# capture_safe_tp_broadcast  -- src forwarded unchanged (Test 5)
# ---------------------------------------------------------------------------


def test_src_forwarded_pynccl_path() -> None:
    """The src argument is forwarded unchanged on the pynccl path."""
    comm = FakePyncclComm(available=True)
    tp_group = FakeTpGroup(pynccl_comm=comm)

    from sglang.srt.speculative.spec_utils import capture_safe_tp_broadcast

    capture_safe_tp_broadcast(tp_group, (_make_sentinel("t"),), src=42)
    assert comm.broadcast_calls[0][1] == 42


def test_src_forwarded_c10d_path() -> None:
    """The src argument is forwarded unchanged on the c10d fallback path."""
    tp_group = FakeTpGroup(pynccl_comm=None)

    from sglang.srt.speculative.spec_utils import capture_safe_tp_broadcast

    capture_safe_tp_broadcast(tp_group, (_make_sentinel("t"),), src=99)
    assert tp_group.broadcast_calls[0][1] == 99


# ---------------------------------------------------------------------------
# capture_safe_tp_broadcast  -- change_state context (Test 6)
# ---------------------------------------------------------------------------


def test_pynccl_change_state_enter_exit() -> None:
    """The pynccl path enters the change_state(enable=True) context and
    exits it correctly."""
    comm = FakePyncclComm(available=True)
    tp_group = FakeTpGroup(pynccl_comm=comm)

    entered = [False]
    exited = [False]

    @contextmanager
    def tracking_change_state(enable: bool = True):
        entered[0] = True
        yield FakeChangeState()
        exited[0] = True

    comm.change_state = tracking_change_state

    from sglang.srt.speculative.spec_utils import capture_safe_tp_broadcast

    capture_safe_tp_broadcast(tp_group, (_make_sentinel("t"),), src=0)

    assert entered[0], "change_state(enable=True) context was never entered"
    assert exited[0], "change_state(enable=True) context was never exited"


# ---------------------------------------------------------------------------
# spec_accept_broadcast_src  -- fast lane off (Test 7)
# ---------------------------------------------------------------------------


def test_spec_accept_broadcast_src_defaults_to_zero() -> None:
    """Returns 0 when the weightless-KV fast lane is off.

    spec_accept_broadcast_src imports get_server_args from
    sglang.srt.runtime_context *inside* the function body, so we patch it
    on the runtime_context module (the import target), not on eagle_utils.
    """
    from unittest.mock import patch

    from sglang.srt.speculative.eagle_utils import spec_accept_broadcast_src

    class FakeServerArgs:
        weightless_kv_fastlane = False
        weightless_kv_head_rank = 5  # should be ignored

    with patch(
        "sglang.srt.runtime_context.get_server_args",
        return_value=FakeServerArgs(),
    ):
        result = spec_accept_broadcast_src()

    assert result == 0, f"expected src=0 when fast lane is off, got {result}"


# ---------------------------------------------------------------------------
# spec_accept_broadcast_src  -- fast lane on (Test 8)
# ---------------------------------------------------------------------------


def test_spec_accept_broadcast_src_returns_head_rank() -> None:
    """Returns the configured head rank when the weightless-KV fast lane
    is on."""
    from unittest.mock import patch

    from sglang.srt.speculative.eagle_utils import spec_accept_broadcast_src

    class FakeServerArgs:
        weightless_kv_fastlane = True
        weightless_kv_head_rank = 3

    with patch(
        "sglang.srt.runtime_context.get_server_args",
        return_value=FakeServerArgs(),
    ):
        result = spec_accept_broadcast_src()

    assert result == 3, f"expected src=3 (head_rank), got {result}"
