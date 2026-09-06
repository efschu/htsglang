"""#1223 hold-at-the-wall: hermetic tests for the hold helper.

No GPU, no torch, no scheduler. The tensor case is covered by a duck-typed
stand-in, which is exactly what ``debug_hold._is_tensor`` matches on -- the
helper deliberately does not import torch (an import inside a crash handler is
a way to lose the crash).
"""

import os
import socket
import threading
import time

import pytest

from sglang.srt.managers import debug_hold


class FakeTensor:
    """Duck-typed tensor: shape/dtype/device/numel, the four attributes the
    summariser keys on. Its repr is deliberately enormous, so a test that sees
    the repr in the dump has caught the summariser not firing."""

    def __init__(self, values, shape, dtype="torch.float32", device="cuda:0"):
        self._values = values
        self.shape = shape
        self.dtype = dtype
        self.device = device

    def numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self._values

    def __repr__(self):
        return "FULL_TENSOR_REPR " + ("9" * 50000)


def _raise_nested(payload_small, payload_big, payload_tensor):
    def inner():
        inner_only_local = "INNER_MARKER"  # noqa: F841 - read out of the dump
        raise RuntimeError("#1223 test wall")

    outer_marker = payload_small  # noqa: F841 - read out of the dump
    big_list = payload_big  # noqa: F841
    slot_votes = payload_tensor  # noqa: F841
    inner()


def _make_exception():
    try:
        _raise_nested(
            "OUTER_MARKER",
            list(range(20000)),
            FakeTensor([0, 3, 3], (3,)),
        )
    except RuntimeError as exc:
        return exc
    raise AssertionError("the fixture must raise")


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for name in (
        debug_hold.HOLD_ENV,
        debug_hold.HOLD_S_ENV,
        debug_hold.HOLD_PORT_BASE_ENV,
        debug_hold.HOLD_INJECT_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(debug_hold.HOLD_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(debug_hold.TAG_ENV, "unittest")


# ---------------------------------------------------------------------------
# The dump
# ---------------------------------------------------------------------------


def test_dump_names_every_frames_locals_and_summarises_tensors(tmp_path):
    exc = _make_exception()
    path = debug_hold.write_dump(exc, rank=2, path=str(tmp_path / "d.txt"))
    text = open(path).read()

    # Both frames of the raising stack, by their variable NAMES -- the half
    # that cannot be reconstructed from a traceback afterwards.
    assert "outer_marker" in text
    assert "OUTER_MARKER" in text
    assert "inner_only_local" in text
    assert "INNER_MARKER" in text

    # The tensor is SUMMARISED, never printed.
    assert "<tensor shape=(3,)" in text
    assert "dtype=torch.float32" in text
    assert "device=cuda:0" in text
    assert "values=[0, 3, 3]" in text
    assert "FULL_TENSOR_REPR" not in text

    # The large list is capped, but its name still appears.
    assert "big_list" in text
    assert "truncated" in text
    assert len(text) < 200_000

    assert "RuntimeError: #1223 test wall" in text
    assert "OTHER PYTHON THREADS" in text


def test_large_tensor_is_not_valued(tmp_path):
    """Shape/dtype/device always; values only when there are few enough."""
    big = FakeTensor(list(range(4096)), (64, 64))
    summary = debug_hold._summarise(big)
    assert "shape=(64, 64)" in summary
    assert "values=" not in summary


# ---------------------------------------------------------------------------
# The socket console
# ---------------------------------------------------------------------------


def _attach(port, script, timeout=15):
    """Talk to the hold's pdb like `nc` would, and return everything it said."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            client = socket.create_connection(("127.0.0.1", port), timeout=5)
            break
        except OSError:
            time.sleep(0.05)
    else:
        raise AssertionError(f"never accepted a connection on {port}")
    with client:
        client.sendall(script.encode())
        client.settimeout(10)
        chunks = []
        while True:
            try:
                data = client.recv(65536)
            except socket.timeout:
                break
            if not data:
                break
            chunks.append(data)
    return b"".join(chunks).decode(errors="replace")


def test_client_can_evaluate_a_frame_local_over_the_socket(monkeypatch):
    port_base = _free_port()
    monkeypatch.setenv(debug_hold.HOLD_ENV, "1")
    monkeypatch.setenv(debug_hold.HOLD_PORT_BASE_ENV, str(port_base))
    monkeypatch.setenv(debug_hold.HOLD_S_ENV, "60")
    exc = _make_exception()

    result = {}

    def run():
        result["held"] = debug_hold.maybe_hold(exc, scheduler=None, pp_rank=0)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    # `up` moves from the raising frame to its caller, where outer_marker lives.
    transcript = _attach(port_base + 0, "p inner_only_local\nup\np outer_marker\nq\n")
    thread.join(timeout=30)

    assert not thread.is_alive(), "the hold did not release on q"
    assert result["held"] is True
    assert "INNER_MARKER" in transcript
    assert "OUTER_MARKER" in transcript


def test_hold_expires_on_its_own_with_nobody_attached(monkeypatch):
    """The timeout is a deadline, not a hint: a forgotten hold ends itself."""
    port_base = _free_port()
    monkeypatch.setenv(debug_hold.HOLD_ENV, "1")
    monkeypatch.setenv(debug_hold.HOLD_PORT_BASE_ENV, str(port_base))
    monkeypatch.setenv(debug_hold.HOLD_S_ENV, "1")
    exc = _make_exception()

    started = time.monotonic()
    assert debug_hold.maybe_hold(exc, scheduler=None, pp_rank=0) is True
    elapsed = time.monotonic() - started
    assert 0.5 < elapsed < 20, f"hold did not honour its 1 s deadline (took {elapsed}s)"


def test_port_is_bound_to_loopback_only(monkeypatch):
    """An unauthenticated console must not be reachable off-box."""
    port_base = _free_port()
    monkeypatch.setenv(debug_hold.HOLD_ENV, "1")
    monkeypatch.setenv(debug_hold.HOLD_PORT_BASE_ENV, str(port_base))
    monkeypatch.setenv(debug_hold.HOLD_S_ENV, "3")
    exc = _make_exception()

    seen = {}
    real_bind = socket.socket.bind

    def spy_bind(self, address):
        seen["address"] = address
        return real_bind(self, address)

    monkeypatch.setattr(socket.socket, "bind", spy_bind)
    debug_hold.maybe_hold(exc, scheduler=None, pp_rank=1)
    assert seen["address"] == ("127.0.0.1", port_base + 1)


# ---------------------------------------------------------------------------
# The rank, read the way the Scheduler actually stores it
# ---------------------------------------------------------------------------


class _PS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Sched:
    def __init__(self, ps):
        self.ps = ps


def test_rank_comes_from_ps_pp_rank_first_then_ps_tp_rank():
    assert debug_hold.resolve_rank(_Sched(_PS(pp_rank=2, tp_rank=0))) == 2
    assert debug_hold.resolve_rank(_Sched(_PS(tp_rank=1))) == 1


def test_rank_never_reads_the_flat_scheduler_attribute():
    """ba2e88fe: `scheduler.pp_rank` does not exist. A helper that reads it
    labels all three ranks identically and the merged log is unusable."""

    class Trap:
        ps = _PS(pp_rank=1)

        def __getattr__(self, name):
            if name in ("pp_rank", "tp_rank"):
                raise AssertionError(f"read the flat Scheduler.{name}")
            raise AttributeError(name)

    assert debug_hold.resolve_rank(Trap()) == 1


def test_rank_falls_back_to_the_callers_locals_when_scheduler_is_none():
    assert debug_hold.resolve_rank(None, 2, 0) == 2
    assert debug_hold.resolve_rank(None, None, 1) == 1


# ---------------------------------------------------------------------------
# OFF-PATH PINS -- the hard contract
# ---------------------------------------------------------------------------


def test_off_path_calls_nothing_at_all(monkeypatch):
    """Without SGLANG_DEBUG_HOLD=1, maybe_hold touches nothing: no dump, no
    socket, no port resolution, no log line. This is the byte-identical-off
    contract, pinned rather than inspected."""
    called = []
    monkeypatch.setattr(
        debug_hold, "_hold", lambda *a, **k: called.append("hold") or True
    )
    monkeypatch.setattr(
        debug_hold, "write_dump", lambda *a, **k: called.append("dump") or ""
    )
    monkeypatch.setattr(
        debug_hold, "dump_path_for", lambda *a, **k: called.append("path") or ""
    )
    monkeypatch.setattr(
        debug_hold, "resolve_rank", lambda *a, **k: called.append("rank") or 0
    )

    def no_sockets(*a, **k):
        raise AssertionError("the off-path opened a socket")

    monkeypatch.setattr(socket, "socket", no_sockets)

    assert debug_hold.maybe_hold(_make_exception(), scheduler=None, pp_rank=0) is False
    assert called == []


@pytest.mark.parametrize("value", ["0", "", "true", "yes", "2"])
def test_only_the_literal_one_arms_the_hold(monkeypatch, value):
    monkeypatch.setenv(debug_hold.HOLD_ENV, value)
    monkeypatch.setattr(
        debug_hold, "_hold", lambda *a, **k: pytest.fail("armed on " + repr(value))
    )
    assert debug_hold.maybe_hold(_make_exception()) is False


# ---------------------------------------------------------------------------
# The inject
# ---------------------------------------------------------------------------


def test_inject_raises_only_for_its_own_marker(monkeypatch):
    monkeypatch.setenv(debug_hold.HOLD_ENV, "1")
    monkeypatch.setenv(debug_hold.HOLD_INJECT_ENV, "cutover")
    debug_hold.maybe_inject("some_other_phase")  # not mine: silent no-op
    with pytest.raises(RuntimeError, match="#1223 INJECTED WALL"):
        debug_hold.maybe_inject("cutover")


def test_inject_refuses_without_the_hold_flag(monkeypatch):
    """An inject with no hold behind it is a boot-killer wearing a debug label."""
    monkeypatch.setenv(debug_hold.HOLD_INJECT_ENV, "cutover")
    debug_hold.maybe_inject("cutover")  # must NOT raise


def test_inject_is_a_noop_with_no_env():
    debug_hold.maybe_inject("cutover")


# ---------------------------------------------------------------------------
# CAN-FAIL PROOFS -- one per direction. These assert that the two tests above
# that carry the load would actually go red against a broken helper, rather
# than passing for an unrelated reason.
# ---------------------------------------------------------------------------


def test_canfail_helper_that_never_opens_the_port_fails_the_socket_test(monkeypatch):
    """Direction 1: no port -> the attach test must fail, not hang forever."""
    port_base = _free_port()
    monkeypatch.setenv(debug_hold.HOLD_ENV, "1")
    monkeypatch.setenv(debug_hold.HOLD_PORT_BASE_ENV, str(port_base))
    monkeypatch.setenv(debug_hold.HOLD_S_ENV, "3")

    def deaf_bind(self, address):
        raise OSError("simulated: this helper never opens the port")

    monkeypatch.setattr(socket.socket, "bind", deaf_bind)
    # The helper reports honestly instead of pretending to hold ...
    assert debug_hold.maybe_hold(_make_exception(), scheduler=None, pp_rank=0) is False
    # ... and the attach the real test performs cannot succeed.
    with pytest.raises(AssertionError, match="never accepted a connection"):
        _attach(port_base + 0, "q\n", timeout=2)


def test_canfail_helper_that_ignores_the_timeout_fails_the_expiry_test(monkeypatch):
    """Direction 2: a hold whose accept() has no deadline never returns, so the
    expiry test's bound is a real bound and not a formality."""
    port_base = _free_port()
    monkeypatch.setenv(debug_hold.HOLD_ENV, "1")
    monkeypatch.setenv(debug_hold.HOLD_PORT_BASE_ENV, str(port_base))
    monkeypatch.setenv(debug_hold.HOLD_S_ENV, "1")

    real_settimeout = socket.socket.settimeout

    def ignore_deadline(self, value):
        # The mutant: honour blocking mode, drop every deadline.
        return real_settimeout(self, None)

    monkeypatch.setattr(socket.socket, "settimeout", ignore_deadline)

    done = threading.Event()

    def run():
        try:
            debug_hold.maybe_hold(_make_exception(), scheduler=None, pp_rank=0)
        finally:
            done.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    # The real helper returns in ~1 s (see test_hold_expires_on_its_own).
    # The mutant is still blocked in accept() well past that.
    assert not done.wait(6), (
        "the timeout-ignoring mutant returned; the bound is vacuous"
    )

    # Release the mutant so the test process does not leak a blocked thread.
    monkeypatch.undo()
    try:
        socket.create_connection(("127.0.0.1", port_base), timeout=5).close()
    except OSError:
        pass
    done.wait(10)


def test_canfail_dump_that_prints_tensors_fails_the_dump_test(tmp_path, monkeypatch):
    """Direction 3: the tensor assertion is load-bearing -- a summariser that
    falls through to repr() puts FULL_TENSOR_REPR in the file."""
    monkeypatch.setattr(debug_hold, "_is_tensor", lambda value: False)
    path = debug_hold.write_dump(
        _make_exception(), rank=0, path=str(tmp_path / "mutant.txt")
    )
    text = open(path).read()
    assert "FULL_TENSOR_REPR" in text, "the mutant should leak the raw repr"
    assert "<tensor shape=" not in text


def test_dump_path_layout(monkeypatch, tmp_path):
    monkeypatch.setenv(debug_hold.HOLD_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(debug_hold.TAG_ENV, "weg1hold1223")
    path = debug_hold.dump_path_for(2)
    assert path.startswith(str(tmp_path))
    assert os.path.basename(path).startswith("weg1hold1223_rank2_")
    assert path.endswith(".txt")
