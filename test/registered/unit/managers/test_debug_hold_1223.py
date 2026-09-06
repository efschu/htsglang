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
        debug_hold.HOLD_INJECT_RANK_ENV,
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


def test_rank_comes_from_ps_when_the_caller_supplied_nothing():
    assert debug_hold.resolve_rank(_Sched(_PS(pp_rank=2, tp_rank=0))) == 2
    assert debug_hold.resolve_rank(_Sched(_PS(tp_rank=1))) == 1


# --- (1) the run-1 defect, as a regression test ----------------------------


def test_cutover_rebound_ps_must_not_override_the_callers_pp_rank():
    """THE weg1holdg1 RUN-1 WALL, reproduced exactly.

    During a pp->tp cutover, phase_flip_runtime.py:3329 rebinds
    `scheduler.ps = replace(boot_ps, tp_rank=world_rank, pp_rank=0, pp_size=1)`
    -- pp_rank is hardcoded 0 on EVERY rank. A wall raised later inside that
    same _cutover (run 1: ReqPoolRebindRefused) therefore sees ps.pp_rank == 0
    on all three ranks. Reading ps first collapsed all three to rank 0, all
    three raced for port 5000, and two lost and died.
    """
    for world_rank, caller_pp in ((0, 0), (1, 1), (2, 2)):
        cutover_ps = _PS(pp_rank=0, pp_size=1, tp_rank=world_rank, tp_size=3)
        rank, source = debug_hold.resolve_rank_and_source(
            _Sched(cutover_ps), caller_pp, 0
        )
        assert rank == caller_pp, (
            f"rank collapsed to {rank} for PP{caller_pp} -- this is run 1"
        )
        assert source == "caller.pp_rank"

    # And the ports must therefore be distinct, which is the property that
    # actually failed on the metal.
    ranks = {
        debug_hold.resolve_rank(
            _Sched(_PS(pp_rank=0, pp_size=1, tp_rank=w, tp_size=3)), w, 0
        )
        for w in range(3)
    }
    assert ranks == {0, 1, 2}


def test_the_boot_drivers_local_patch_order_would_still_collapse():
    """Recorded so the rejected fix is not re-proposed: the driver's order
    (ps.pp_rank -> caller pp -> ps.tp_rank -> caller tp) still lets ps.pp_rank
    answer 0 first at a cutover wall. It passed run 2 only because run 2's wall
    was outside a cutover."""

    def driver_order(ps, caller_pp, caller_tp):
        for v in (
            getattr(ps, "pp_rank", None),
            caller_pp,
            getattr(ps, "tp_rank", None),
            caller_tp,
        ):
            if isinstance(v, int):
                return v
        return 0

    cutover_ps = _PS(pp_rank=0, pp_size=1, tp_rank=1, tp_size=3)
    assert driver_order(cutover_ps, 1, 0) == 0  # the defect, still present
    assert debug_hold.resolve_rank(_Sched(cutover_ps), 1, 0) == 1  # ours is fixed


def test_rank_source_is_reported():
    assert debug_hold.resolve_rank_and_source(None, 2, 0)[1] == "caller.pp_rank"
    assert debug_hold.resolve_rank_and_source(None, None, 1)[1] == "caller.tp_rank"
    assert (
        debug_hold.resolve_rank_and_source(_Sched(_PS(pp_rank=2)), None, None)[1]
        == "ps.pp_rank"
    )
    assert debug_hold.resolve_rank_and_source(None, None, None) == (0, "default")


def test_a_bool_is_not_a_rank():
    """bool is an int subclass; True would silently pass as rank 1."""
    assert debug_hold.resolve_rank_and_source(None, True, 2) == (2, "caller.tp_rank")


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


# --- (2) a bind failure must never cost the hold ---------------------------


def test_port_collision_falls_back_to_an_ephemeral_port_and_still_holds(monkeypatch):
    """weg1holdg1 run 1: two ranks lost the race for port 5000, did NOT hold,
    and their death path tore down the rank that was holding correctly."""
    busy = socket.socket()
    busy.bind(("127.0.0.1", 0))
    busy.listen(1)
    taken = busy.getsockname()[1]

    monkeypatch.setenv(debug_hold.HOLD_ENV, "1")
    monkeypatch.setenv(debug_hold.HOLD_PORT_BASE_ENV, str(taken))
    monkeypatch.setenv(debug_hold.HOLD_S_ENV, "20")
    exc = _make_exception()

    bound = []
    real_bind = socket.socket.bind

    def spy_bind(self, address):
        real_bind(self, address)
        bound.append(self.getsockname())

    monkeypatch.setattr(socket.socket, "bind", spy_bind)

    result = {}

    def run():
        result["held"] = debug_hold.maybe_hold(exc, scheduler=None, pp_rank=0)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    time.sleep(2)
    # It must have bound SOMETHING despite the collision, and not the busy port.
    assert bound, "no successful bind at all"
    actual_port = bound[-1][1]
    assert actual_port != taken
    transcript = _attach(actual_port, "p inner_only_local\nq\n")
    thread.join(timeout=30)
    busy.close()

    assert result["held"] is True, "a port collision must not cancel the hold"
    assert "INNER_MARKER" in transcript


def test_hold_mode_suppresses_the_sigquit_in_the_scheduler_except():
    """The other half of run 1's damage: a non-holding rank reaching the normal
    death path SIGQUITs the parent, which ends the peers that ARE held.

    Pins the guard's shape in the source rather than booting a scheduler."""
    import pathlib

    src = pathlib.Path(debug_hold.__file__).parent / "scheduler.py"
    text = src.read_text()
    idx = text.index("parent_process.send_signal(signal.SIGQUIT)", text.index("#1223"))
    window = text[idx - 2000 : idx + 400]
    assert "_1223_hold_on" in window, "the SIGQUIT is not guarded by the hold flag"
    assert "if _1223_hold_on:" in window
    assert "else:\n            parent_process.send_signal(signal.SIGQUIT)" in window
    assert "if not _1223_hold_on and envs.SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION" in text


# --- (3) the dump filename must not collide --------------------------------


def test_dump_filenames_of_two_ranks_in_the_same_second_differ(monkeypatch, tmp_path):
    """weg1holdg1 run 1: PP2's dump silently OVERWROTE PP1's, because both
    resolved rank 0 in the same second. The name must be unique even when the
    rank resolution is wrong -- that is precisely the case that matters."""
    monkeypatch.setenv(debug_hold.HOLD_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(debug_hold.TAG_ENV, "weg1holdg1")

    real_pid = os.getpid()
    a = debug_hold.dump_path_for(0, port=5000)
    # A second PROCESS, same (wrongly resolved) rank, same second.
    monkeypatch.setattr(os, "getpid", lambda: real_pid + 1)
    b = debug_hold.dump_path_for(0, port=5000)
    assert a != b, "two processes, same rank, same second -> same file"
    assert f"pid{real_pid}" in a
    assert f"pid{real_pid + 1}" in b
    assert "port5000" in a


def test_dump_filename_carries_pid_and_port(monkeypatch, tmp_path):
    monkeypatch.setenv(debug_hold.HOLD_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(debug_hold.TAG_ENV, "weg1hold1223")
    name = os.path.basename(debug_hold.dump_path_for(2, port=5002))
    assert name.startswith("weg1hold1223_rank2_")
    assert f"_pid{os.getpid()}_" in name
    assert "_port5002_" in name
    assert name.endswith(".txt")


# --- (4) sessions: only q releases; EOF keeps holding ----------------------


def test_disconnect_without_q_keeps_the_rank_held(monkeypatch):
    """A scripted `printf 'w\\np x\\n' | nc` closes the pipe when it is done.
    That EOF used to release the rank after ONE block, which is why the boot
    driver needed a FIFO. A disconnect is not a decision."""
    port_base = _free_port()
    monkeypatch.setenv(debug_hold.HOLD_ENV, "1")
    monkeypatch.setenv(debug_hold.HOLD_PORT_BASE_ENV, str(port_base))
    monkeypatch.setenv(debug_hold.HOLD_S_ENV, "45")
    exc = _make_exception()

    result = {}

    def run():
        result["held"] = debug_hold.maybe_hold(exc, scheduler=None, pp_rank=0)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    # Session 1: no q, just a query and a disconnect.
    first = _attach(port_base, "p inner_only_local\n")
    assert "INNER_MARKER" in first
    thread.join(timeout=5)
    assert thread.is_alive(), "EOF released the hold -- it must keep holding"

    # Session 2 proves it is still accepting, and q ends it.
    second = _attach(port_base, "up\np outer_marker\nq\n")
    assert "OUTER_MARKER" in second
    thread.join(timeout=30)
    assert not thread.is_alive(), "q did not release the hold"
    assert result["held"] is True


def test_scripted_one_shot_attach_pattern_works_repeatedly(monkeypatch):
    """The documented scripted pattern: several independent piped sessions,
    then a final one carrying q."""
    port_base = _free_port()
    monkeypatch.setenv(debug_hold.HOLD_ENV, "1")
    monkeypatch.setenv(debug_hold.HOLD_PORT_BASE_ENV, str(port_base))
    monkeypatch.setenv(debug_hold.HOLD_S_ENV, "60")
    exc = _make_exception()

    thread = threading.Thread(
        target=lambda: debug_hold.maybe_hold(exc, scheduler=None, pp_rank=0),
        daemon=True,
    )
    thread.start()
    for _ in range(3):
        assert "INNER_MARKER" in _attach(port_base, "p inner_only_local\n")
        assert thread.is_alive()
    _attach(port_base, "q\n")
    thread.join(timeout=30)
    assert not thread.is_alive()


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


# --- #1225: the two new inject markers -------------------------------------


@pytest.mark.parametrize("marker", ["cutover", "last_chunk", "abandon"])
def test_each_marker_fires_only_for_itself(monkeypatch, marker):
    monkeypatch.setenv(debug_hold.HOLD_ENV, "1")
    monkeypatch.setenv(debug_hold.HOLD_INJECT_ENV, marker)
    with pytest.raises(RuntimeError, match="#1223 INJECTED WALL"):
        debug_hold.maybe_inject(marker, pp_rank=1)
    for other in debug_hold.KNOWN_INJECT_MARKERS:
        if other != marker:
            debug_hold.maybe_inject(other, pp_rank=1)  # must not raise


@pytest.mark.parametrize("marker", ["last_chunk", "abandon"])
def test_new_markers_refuse_without_the_hold_flag(monkeypatch, marker):
    monkeypatch.setenv(debug_hold.HOLD_INJECT_ENV, marker)
    debug_hold.maybe_inject(marker, pp_rank=1)  # must NOT raise


def test_inject_rank_filter_selects_one_follower(monkeypatch):
    """#1225 INJECT-1 must hold ONE follower and leave the others running --
    the analysis is a cross-rank diff, so holding all three at the first site
    would destroy the comparison the boot exists to make."""
    monkeypatch.setenv(debug_hold.HOLD_ENV, "1")
    monkeypatch.setenv(debug_hold.HOLD_INJECT_ENV, "last_chunk")
    monkeypatch.setenv(debug_hold.HOLD_INJECT_RANK_ENV, "1")

    with pytest.raises(RuntimeError, match="#1223 INJECTED WALL"):
        debug_hold.maybe_inject("last_chunk", pp_rank=1)
    for free_rank in (0, 2):
        debug_hold.maybe_inject("last_chunk", pp_rank=free_rank)  # must not raise


def test_inject_rank_unset_means_every_rank(monkeypatch):
    monkeypatch.setenv(debug_hold.HOLD_ENV, "1")
    monkeypatch.setenv(debug_hold.HOLD_INJECT_ENV, "abandon")
    monkeypatch.delenv(debug_hold.HOLD_INJECT_RANK_ENV, raising=False)
    for rank in (0, 1, 2):
        with pytest.raises(RuntimeError, match="#1223 INJECTED WALL"):
            debug_hold.maybe_inject("abandon", pp_rank=rank)


def test_inject_rank_filter_uses_the_boot_rank_not_the_rebound_ps(monkeypatch):
    """The filter must not be fooled by the cutover's ps rebind either: with
    ps.pp_rank == 0 on every rank, a filter reading ps would hold rank 0 and
    silently never fire for the follower that was asked for."""
    monkeypatch.setenv(debug_hold.HOLD_ENV, "1")
    monkeypatch.setenv(debug_hold.HOLD_INJECT_ENV, "abandon")
    monkeypatch.setenv(debug_hold.HOLD_INJECT_RANK_ENV, "2")
    cutover_ps = _PS(pp_rank=0, pp_size=1, tp_rank=2, tp_size=3)
    with pytest.raises(RuntimeError, match="#1223 INJECTED WALL"):
        debug_hold.maybe_inject("abandon", _Sched(cutover_ps), pp_rank=2)


def test_inject_sites_are_wired_at_the_named_places():
    """Delivery is code presence PLUS a caller: pin that both sites exist and,
    for INJECT-1, that it sits OUTSIDE the instrument's swallowing except --
    a raise inside it is eaten and the wall silently never happens."""
    import pathlib

    mgr = pathlib.Path(debug_hold.__file__).parent
    pfr = (mgr / "phase_flip_runtime.py").read_text()
    sched = (mgr / "scheduler.py").read_text()

    # INJECT-2 at the abandon branch ENTRY, before the ledger clear.
    entry = pfr.index("if reduced_fit[0] == 0 or not frames_agree:")
    clear = pfr.index("self._armed_residents = {}  # #1202: and so is the ledger")
    inject2 = pfr.index('maybe_inject("abandon"', entry)
    assert entry < inject2 < clear, "INJECT-2 is not before the ledger clear"

    # INJECT-1 outside the swallowing except of _trace_pp_admission_verdict.
    swallow = sched.index('"#788 PP-ADMISSION trace unavailable: %s: %s"')
    inject1 = sched.index('maybe_inject(\n                "last_chunk"')
    assert inject1 > swallow, "INJECT-1 is inside the except that swallows it"
    assert "_1223_last_chunk = ret is not None and chunked == 0" in sched

    assert 'maybe_inject("cutover", pp_rank=world_rank)' in pfr


# ---------------------------------------------------------------------------
# CAN-FAIL PROOFS -- one per direction. These assert that the two tests above
# that carry the load would actually go red against a broken helper, rather
# than passing for an unrelated reason.
# ---------------------------------------------------------------------------


def test_canfail_helper_that_never_opens_the_port_fails_the_socket_test(monkeypatch):
    """Direction 1: no port -> every attach test must FAIL rather than hang.

    Note the contract change from the pre-metal version: a helper that cannot
    bind now still HOLDS (run 1 proved that returning early kills the peers),
    so the can-fail is about the ATTACH being impossible, not about the return
    value."""
    port_base = _free_port()
    monkeypatch.setenv(debug_hold.HOLD_ENV, "1")
    monkeypatch.setenv(debug_hold.HOLD_PORT_BASE_ENV, str(port_base))
    monkeypatch.setenv(debug_hold.HOLD_S_ENV, "3")

    def deaf_bind(self, address):
        raise OSError("simulated: this helper never opens the port")

    monkeypatch.setattr(socket.socket, "bind", deaf_bind)
    started = time.monotonic()
    # It still holds (the peers must survive) ...
    assert debug_hold.maybe_hold(_make_exception(), scheduler=None, pp_rank=0) is True
    # ... for the full deadline, and then returns on its own.
    assert 2 < time.monotonic() - started < 25
    # ... and the attach that every socket test performs cannot succeed.
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


def test_dump_path_layout_legacy(monkeypatch, tmp_path):
    monkeypatch.setenv(debug_hold.HOLD_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(debug_hold.TAG_ENV, "weg1hold1223")
    path = debug_hold.dump_path_for(2)
    assert path.startswith(str(tmp_path))
    assert os.path.basename(path).startswith("weg1hold1223_rank2_")
    assert path.endswith(".txt")
