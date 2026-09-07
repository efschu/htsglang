# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#1233 weg2 S2 -- BI-1 ``GroupCoordinator.barlink_reopen()``.

Weg 2 puts one of two process groups to sleep on every card. Sleep closes that
group's barlink transports (``BarlinkBar1Transport.close()``, which is
collective-free and gives the BAR1 aperture back); wake has to build them again.
Today there is exactly one build site -- the block inside
``GroupCoordinator.__init__`` -- and no way to reach it a second time, so a woken
group would have no transports and no abort-gate registration.

RED-FIRST against aef3ae7676, where neither ``_build_barlink`` nor
``barlink_reopen`` exists.

WHAT IS PINNED, AND WHY EACH ONE CAN FAIL

(a) The construction block is ONE definition, called from two places. If the
    reopen path grew its own copy of the block, the two would drift and only the
    boot path would keep the abort-gate registration, the ledger credit and the
    achieved-vs-requested log line. That is the second-bookkeeping shape the
    upstream-minimal law targets, so the test asserts the gate literal lives in
    ``_build_barlink`` and that ``__init__`` only calls it.

(b) The behaviour the slice exists for, hermetically: build -> close -> a
    collective refuses -> reopen -> the collective succeeds again, with the BAR1
    ledger back at its exact prior value and exactly one abort-gate registration.
    The transports themselves need three cards and are proven on the slice boot;
    what is provable at the desk is the WIRING -- that reopen goes through the
    same construction, hence re-registers and re-credits, and that the ORDER is
    close-then-build, which the fake models by refusing a second credit for a
    group that still holds one.

    The refusal in that sequence is the FAKE's, so it proves the sequence, not
    the tree. The tree's own refusal is pinned separately, on the real class,
    by ``test_a_closed_communicator_refuses_instead_of_the_gloo_plane``: before
    this slice ``BarlinkCommunicator.close()`` left ``self.transport`` in place
    and every collective went on ANSWERING over the host-staged gloo plane
    (``_select`` returns None for a down transport, and None means gloo), so a
    group whose wake never called ``barlink_reopen()`` served without one line
    saying it had left bar1.

(c) The danger direction, which is the reason the spec forbids ``destroy()`` as
    a sleep: ``GroupCoordinator.destroy()`` also tears down the gloo
    ``cpu_group`` (``parallel_state.py:2623-2626``), and every barlink bring-up
    needs it -- the fd exchange (``barlink_bar1.py:1534 _exchange_fds``), the
    liveness install, the window minimum all run collectives on it. A reopen
    after a ``destroy()`` must refuse by name at the top, not fault somewhere
    inside the fd exchange.

GREEN PINS (green before this slice and after it) guard the three things S2
promises NOT to change: the flag gate, ``destroy()``, and ``close()``.
"""

import ast
import inspect
import pathlib

import pytest
import torch

from sglang.srt.distributed import parallel_state as ps
from sglang.srt.distributed.device_communicators import barlink as barlink_mod
from sglang.srt.distributed.device_communicators import barlink_abort_gate as gate
from sglang.srt.distributed.device_communicators import (
    barlink_matrix_transport as ledger,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PARALLEL_STATE = pathlib.Path(inspect.getsourcefile(ps))
_BAR1 = pathlib.Path(
    inspect.getsourcefile(
        __import__(
            "sglang.srt.distributed.device_communicators.barlink_bar1",
            fromlist=["x"],
        )
    )
)

_GATE_LITERAL = "if should_build_barlink(self.world_size):"

# An arbitrary but fixed region size for the fake credit. The number is never
# compared against a card; only "back to the prior value" is asserted, so its
# denominator is the ledger's own list, not an aperture.
_FAKE_REGION_BYTES = 24 * 1024 * 1024


def _fn(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is not defined in {_PARALLEL_STATE}")


def _tree() -> ast.AST:
    return ast.parse(_PARALLEL_STATE.read_text())


class _FakeComm:
    """Stands in for ``BarlinkCommunicator`` with the two facts S2 needs.

    It takes the ledger credit and the abort-gate registration that a real
    bring-up takes, and it refuses collectives once closed. Everything else a
    real communicator does needs cards.

    It also models ONE metal constraint, because without it the close-first
    order is unobservable: a real build prices its window against
    ``bar1_free - reserve`` while the old transport's pages are still mapped
    (``barlink_matrix_transport.py:345``, and the comment above it says NVML
    free "already includes what this process has pinned"). So a second build
    for a group whose credit still stands is refused here, and the build/close
    sequence is recorded in ``events`` for the test that reads it.
    """

    built = 0
    events: list = []

    def __init__(self, cpu_group, device, group):
        assert cpu_group is not None, "a barlink build without a cpu_group"
        assert all(g != group for g, _ in ledger.ledger_balance(device)), (
            f"a second BAR1 window for group {group!r} while the first credit "
            "still stands: window_for would price the new window against "
            "space this process has not handed back"
        )
        type(self).events.append("build")
        self.cpu_group = cpu_group
        self.device = device
        self.group = group
        self.closed = False
        self.state = {"direct": True, "achieved": "bar1"}
        type(self).built += 1
        ledger.ledger_credit(device, group, _FAKE_REGION_BYTES)
        gate.register(self)

    def captured_launches(self) -> bool:
        """Delegates to the REAL module function, so this stays a fake of the
        wiring and not a fake of the answer."""
        return barlink_mod.transport_captured_launches(self)

    def all_reduce(self, value):
        if self.closed:
            raise RuntimeError(
                f"barlink group {self.group!r}: the transport is closed; "
                "no collective can run until it is reopened"
            )
        return value

    def close(self):
        if self.closed:
            return
        self.closed = True
        type(self).events.append("close")
        gate.unregister(self)
        ledger.ledger_debit(self.device, self.group)


@pytest.fixture
def coord(monkeypatch):
    """A GroupCoordinator carcass with exactly the four attributes the barlink
    construction block reads, and a faked communicator class."""
    monkeypatch.setenv("SGLANG_BARLINK_TRANSPORT", "device")
    monkeypatch.setattr(ps, "should_build_barlink", lambda world_size: True)
    monkeypatch.setattr(barlink_mod, "BarlinkCommunicator", _FakeComm)
    _FakeComm.built = 0
    _FakeComm.events = []
    gate.reset_for_test()

    c = ps.GroupCoordinator.__new__(ps.GroupCoordinator)
    c.world_size = 3
    c.cpu_group = object()
    c.device = torch.device("cuda:0")
    c.unique_name = "tp:0"
    c.barlink_comm = None
    # The BAR1 ledger is process-global. A credit left on this ordinal by
    # anything else would make _FakeComm's "no second window for a live
    # group" assertion fire for the wrong reason, so the fixture starts from
    # an empty one -- and a leftover credit in a CPU-only run is a leak in
    # its own right.
    for _group, _ in ledger.ledger_balance(c.device):
        ledger.ledger_debit(c.device, _group)
    yield c
    comm = c.barlink_comm
    if comm is not None:
        comm.close()
    gate.reset_for_test()


# ---------------------------------------------------------------------------
# RED
# ---------------------------------------------------------------------------


def test_the_construction_block_is_one_named_method():
    """(a) The gate literal moves into ``_build_barlink`` and nowhere else."""
    assert hasattr(ps.GroupCoordinator, "_build_barlink"), (
        "GroupCoordinator._build_barlink is missing -- the barlink construction "
        "block has only one call site and cannot be reached at wake"
    )
    body = ast.unparse(_fn(_tree(), "_build_barlink"))
    assert "should_build_barlink(self.world_size)" in body, (
        "_build_barlink must CARRY the flag gate, not sit beside it"
    )
    src = _PARALLEL_STATE.read_text()
    assert src.count(_GATE_LITERAL) == 1, (
        f"{_GATE_LITERAL!r} occurs {src.count(_GATE_LITERAL)} times -- a second "
        "copy of the construction block is the second-bookkeeping shape"
    )


def test_init_delegates_instead_of_inlining():
    """(a) ``__init__`` calls the method; the block is no longer inlined."""
    init = ast.unparse(_fn(_tree(), "__init__"))
    assert "self._build_barlink()" in init, (
        "GroupCoordinator.__init__ must build barlink through _build_barlink()"
    )
    assert "should_build_barlink(self.world_size)" not in init, (
        "the construction block is still inlined in __init__ -- extracting it "
        "is the whole point, and a leftover copy would drift"
    )


def test_reopen_exists_and_never_destroys():
    """(c) ``barlink_reopen`` exists and does not go through ``destroy()``."""
    assert hasattr(ps.GroupCoordinator, "barlink_reopen"), (
        "GroupCoordinator.barlink_reopen is missing -- a woken group has no transports"
    )
    src = inspect.getsource(ps.GroupCoordinator.barlink_reopen)
    body = ast.unparse(_fn(ast.parse(src.lstrip()), "barlink_reopen"))
    for forbidden in ("self.destroy(", "destroy_process_group"):
        assert forbidden not in body, (
            f"barlink_reopen must never reach {forbidden!r}: destroy() also "
            "tears down the gloo cpu_group that the rebuild needs "
            "(parallel_state.py:2623-2626)"
        )


def test_close_then_collective_refuses_then_reopen_restores(coord):
    """(b) The sequence the slice exists for, wired end to end."""
    coord._build_barlink()
    comm = coord.barlink_comm
    assert comm is not None
    assert _FakeComm.built == 1
    prior_ledger = ledger.ledger_balance(coord.device)
    assert len(prior_ledger) == 1
    assert len(gate.registered()) == 1
    assert comm.all_reduce(7) == 7

    # --- sleep: the transport half only. close() is left unchanged.
    comm.close()
    with pytest.raises(RuntimeError, match="closed"):
        comm.all_reduce(7)
    assert ledger.ledger_balance(coord.device) == [], (
        "the BAR1 ledger must return to zero after close, or the next build's "
        "window_for subtracts space nobody holds"
    )
    assert gate.registered() == [], "a closed transport must leave the abort gate"

    # --- wake
    coord.barlink_reopen()
    reopened = coord.barlink_comm
    assert reopened is not None
    assert reopened is not comm, "reopen must build a NEW communicator"
    assert _FakeComm.built == 2
    assert reopened.all_reduce(7) == 7
    assert ledger.ledger_balance(coord.device) == prior_ledger, (
        "the BAR1 ledger must come back to its exact prior value after reopen"
    )
    assert gate.registered() == [reopened], (
        "the abort gate must hold exactly the reopened transport -- a reopen "
        "without a re-registration leaves the group with no abort poller"
    )


def test_reopen_on_a_live_comm_does_not_double_charge_the_ledger(coord):
    """(b) The other order: a reopen that finds the transports still up.

    `window_for` sizes the new region against `bar1_free - reserve`, and the
    ledger is what tells it how much of that this process already holds. A
    reopen that left the old credit standing would either shrink the new
    window or raise `Bar1WindowRefused` (W6) against space the process itself
    is failing to return -- a refusal with the wrong cause on it. So the
    reopen returns the old one first, and the count is the proof.
    """
    coord._build_barlink()
    first = coord.barlink_comm
    prior_ledger = ledger.ledger_balance(coord.device)

    coord.barlink_reopen()  # no close() in between -- deliberately

    assert first.closed, "reopen must return a still-live transport first"
    assert coord.barlink_comm is not first
    assert _FakeComm.events == ["build", "close", "build"], (
        "the ORDER is the property, not the end state: build-then-close "
        f"reaches the same balance and the same gate list (events: "
        f"{_FakeComm.events}). On metal it means two mapped BAR1 windows for "
        "one group at the same instant, so window_for prices the new one "
        "against space this process has not returned -- a clip, or a W6 with "
        "the wrong cause on it"
    )
    assert ledger.ledger_balance(coord.device) == prior_ledger, (
        "two live BAR1 credits for one group: the next window_for would be "
        "priced against space this process has not given back"
    )
    assert gate.registered() == [coord.barlink_comm], (
        "the abort gate must hold exactly one transport per group"
    )


def test_a_failing_rebuild_propagates_and_leaves_a_refusing_group(coord, monkeypatch):
    """(b) A wake that cannot rebuild must raise -- and REFUSE, not fall to NCCL.

    `_build_barlink` genuinely can raise at wake: `Bar1WindowRefused` (W6)
    when a sibling still holds the aperture, `Bar1Failed` from the holder,
    `_enforce_cpu_transport_needs_eager` on a host transport under graphs.
    Swallowing it would leave THIS rank awake while its peers hold transports.

    But leaving `barlink_comm` None is not a stop either. The dispatch seams
    read exactly `if self.barlink_comm is not None:` (`parallel_state.py:1350`
    for all_reduce) and the terminal branch of that method is
    `inplace_all_reduce(...)` (`:1416`) -- NCCL, because pynccl was never
    built when barlink was active at boot (`_barlink_active`, `:836`). So None
    does not halt the group, it silently moves it to the plane the barlink
    standard forbids; and on three ranks a BAR1 refusal is a PER-CARD fact, so
    one rank can land there while its siblings rebuild. Ranks disagreeing
    about the transport is a hang, not a STOP.

    A failed reopen therefore puts the group back exactly as it found it. The
    old communicator is CLOSED (`close()` sets `_closed` before its early
    return), so every seam that dispatches afterwards hits the named refusal
    in `_select`, while the exception still propagates for the caller's
    verdict: W4, group-fatal.
    """
    coord._build_barlink()
    live = coord.barlink_comm
    assert live is not None

    class _RefusingComm:
        def __init__(self, cpu_group, device, group):
            raise RuntimeError("window refused: no BAR1 aperture left")

    monkeypatch.setattr(barlink_mod, "BarlinkCommunicator", _RefusingComm)

    with pytest.raises(RuntimeError, match="window refused"):
        coord.barlink_reopen()

    assert coord.barlink_comm is live, (
        "a failed wake must leave the group as it found it: the CLOSED "
        "communicator stays installed so the seam predicate "
        "'self.barlink_comm is not None' is still true and the next "
        "collective refuses, instead of falling through to NCCL unannounced"
    )
    assert live.closed, "the old transport is returned before the build"
    with pytest.raises(RuntimeError, match="closed"):
        coord.barlink_comm.all_reduce(7)
    assert ledger.ledger_balance(coord.device) == [], (
        "a failed wake must not leave a BAR1 credit standing"
    )
    assert gate.registered() == [], (
        "a failed wake must not leave a stale abort-gate registration"
    )


def test_reopen_with_the_flag_off_builds_nothing(coord, monkeypatch):
    """The docstring's flag-off claim, with a can-fail proof of its own.

    `barlink_reopen()` says it is "a no-op that leaves `barlink_comm` None,
    exactly as at boot" when the flag is off. Every other behavioural test
    here runs flag-ON (the fixture pins `should_build_barlink` to True), so
    without this one a wake that BUILT barlink where boot did not would pass
    the whole file -- a behaviour change on the default path, which the
    flag-off byte-identity rule forbids.
    """
    monkeypatch.setattr(ps, "should_build_barlink", lambda world_size: False)

    coord.barlink_reopen()

    assert coord.barlink_comm is None, (
        "flag off: the wake must leave barlink_comm None, exactly as boot does"
    )
    assert _FakeComm.built == 0, "flag off: nothing may be constructed"
    assert gate.registered() == []
    assert ledger.ledger_balance(coord.device) == []


def test_a_closed_communicator_refuses_instead_of_the_gloo_plane():
    """(b) on the REAL class: closed means refuse, not answer over gloo.

    `_select` returns None for a transport that declines a size, and None
    means the host-staged gloo plane. For a CLOSED transport that answer is
    wrong: sleep hands the BAR1 aperture back, so a group whose wake never
    called `barlink_reopen()` would keep serving over host staging with no
    log line saying it left bar1 -- the mixed-measurement shape the
    achieved-vs-requested pair exists to prevent. A carcass is used because a
    real bring-up needs cards; the two attributes `_select` reads are set by
    hand.
    """
    comm = barlink_mod.BarlinkCommunicator.__new__(barlink_mod.BarlinkCommunicator)
    comm.transport = None
    comm.group = "tp:0"
    comm._closed = False

    # Open, with no transport: None, i.e. "take the gloo plane". This is the
    # can-fail half -- the refusal below has to be caused by the close.
    assert comm._select("all_reduce", 4096) is None

    barlink_mod.BarlinkCommunicator.close(comm)

    assert comm._closed is True
    with pytest.raises(RuntimeError, match="closed"):
        comm._select("all_reduce", 4096)


def test_reopen_refuses_when_the_cpu_group_is_gone(coord):
    """(c) ``destroy()`` used as a sleep must be refused, by name, at the top."""
    coord._build_barlink()
    coord.barlink_comm.close()
    coord.barlink_comm = None
    coord.cpu_group = None  # what destroy() leaves behind
    with pytest.raises(RuntimeError, match="cpu_group"):
        coord.barlink_reopen()


# ---------------------------------------------------------------------------
# GREEN PINS -- these must be green before this slice and after it
# ---------------------------------------------------------------------------


def test_pin_flag_gate_is_unchanged():
    """The predicate stays the shared one, and the import stays inside it."""
    assert (
        "return bool(envs.SGLANG_BARLINK.get()) and world_size > 1"
        in inspect.getsource(ps.should_build_barlink)
    ), "should_build_barlink must still be exactly the flag-and-multi-rank gate"
    src = _PARALLEL_STATE.read_text()
    import_line = "from sglang.srt.distributed.device_communicators.barlink import"
    assert src.count(import_line) == 1
    assert src.index(import_line) > src.index(_GATE_LITERAL), (
        "the barlink import must sit inside the flag gate -- flag off must not "
        "import the communicator at all"
    )
    # Round 2: the wake's captured-graph question is asked THROUGH the
    # communicator (`old_comm.captured_launches()`), precisely so that it adds
    # no second import here. If it ever grows one, this pin and
    # test_barlink_port.py::test_construction_is_flag_gated both go red.
    assert "captured_launches()" in ast.unparse(_fn(_tree(), "barlink_reopen"))


def test_pin_destroy_still_tears_down_the_cpu_group():
    """The fact that makes destroy() unusable as a sleep must stay true."""
    body = ast.unparse(_fn(_tree(), "destroy"))
    assert "self.barlink_comm.close()" in body
    assert "torch.distributed.destroy_process_group(self.cpu_group)" in body
    assert "self.cpu_group = None" in body


def test_pin_transport_close_is_unchanged():
    """close() stands the abort poller down BEFORE it drops the peers."""
    src = _BAR1.read_text()
    marker = "            barlink_abort_gate.unregister(self)"
    assert src.count(marker) == 1
    assert src.index(marker) < src.index("        self._peers.clear()"), (
        "close() must unregister from the abort gate before the peers go, or "
        "the watchdog holds a reference into a torn-down window"
    )


# ---------------------------------------------------------------------------
# ROUND 2 -- the two remaining silent-wrongs on the wake path, and the shape
# the round-1 refusal test did not cover
# ---------------------------------------------------------------------------


class _Bar1Stub:
    """The one fact ``BarlinkBar1Transport`` carries about graph capture.

    ``_captured_launches`` (``barlink_bar1.py:1798`` init, ``:4793`` set) is
    armed the moment a launch runs under ``graph_capture_running()``. The
    comment at ``:4774-4776`` says what it means: "this transport's kernels
    are now inside a graph and will run on every replay with no host code
    between them".
    """

    def __init__(self, captured: bool):
        self._captured_launches = captured
        self._up = True

    def close(self) -> None:
        self._up = False


class _MatrixStub:
    """Mirrors ``BarlinkMatrixTransport``: ``close()`` NULLS ``bar1``.

    ``barlink_matrix_transport.py:600-603``. That one statement erases the
    only place the capture latch lives, which is why the communicator has to
    snapshot the fact at close time. Without the snapshot the refusal could
    only ever fire on a reopen of a STILL-OPEN transport -- and the shape the
    S2 slice boot drives (close, then reopen) is the other one. That is the
    same defect class as the round-1 review's surviving mutant M2: a proof
    taken on a shape production does not have.
    """

    def __init__(self, captured: bool):
        self.bar1 = _Bar1Stub(captured)

    def handles(self, op: str, nbytes: int) -> bool:
        return self.bar1 is not None and self.bar1._up

    def close(self) -> None:
        if self.bar1 is not None:
            self.bar1.close()
            self.bar1 = None


class _DownTransport:
    """A closed BAR1 transport as ``close()`` actually leaves one.

    ``BarlinkBar1Transport.close()`` sets ``self._up = False`` as its first
    statement (``barlink_bar1.py:5721``) and ``handles()`` reads it
    (``:3128``), so the object stays in place and answers False. It is
    ``BarlinkCommunicator.close()`` that never nulls ``self.transport`` --
    ``self.transport =`` occurs exactly once in ``barlink.py`` (``:702``, in
    ``__init__``). So the real post-sleep shape is ``_closed`` True AND
    ``transport`` NOT None.
    """

    def handles(self, op: str, nbytes: int) -> bool:
        return False

    def close(self) -> None:
        return None


def test_a_closed_communicator_refuses_on_the_shape_close_actually_leaves():
    """The round-1 refusal, proven on the shape production has.

    Its sibling above builds its carcass with ``transport = None``, and both
    halves of that can-fail pair run on that one shape. A refusal narrowed to
    ``_closed and self.transport is None`` therefore passes the whole file
    while a really-slept communicator (``_closed`` True, ``transport`` a
    down BAR1 transport) goes back to answering ``None`` from ``_select`` --
    and ``None`` means the host-staged gloo plane, silently. This pins the
    other shape.
    """
    comm = barlink_mod.BarlinkCommunicator.__new__(barlink_mod.BarlinkCommunicator)
    comm.group = "tp:0"
    comm._closed = False
    comm.transport = _DownTransport()

    # Can-fail half: a transport that merely DECLINES this size is None, i.e.
    # the gloo plane, and that is right. The refusal must come from the close.
    assert comm._select("all_reduce", 4096) is None

    barlink_mod.BarlinkCommunicator.close(comm)

    assert comm.transport is not None, (
        "the shape being proven: close() leaves the transport object in place "
        "(barlink.py assigns self.transport exactly once, at :702), so a "
        "refusal keyed on 'transport is None' would never fire after a sleep"
    )
    with pytest.raises(RuntimeError, match="closed"):
        comm._select("all_reduce", 4096)


def test_the_real_communicator_answers_the_capture_question():
    """The seam `barlink_reopen()` calls must exist on the REAL class.

    The wake asks `old_comm.captured_launches()` with no getattr default: a
    real communicator that lost the method raises AttributeError at the wake
    rather than skipping the refusal, and this pin catches it at the desk.
    """
    assert hasattr(barlink_mod.BarlinkCommunicator, "captured_launches")
    assert "transport_captured_launches(self)" in inspect.getsource(
        barlink_mod.BarlinkCommunicator.captured_launches
    ), "the method must be the seam onto the module function, not a second copy"


def test_close_remembers_that_graphs_held_the_transport_kernels():
    """The capture latch has to survive the teardown that erases it.

    ``BarlinkMatrixTransport.close()`` nulls ``self.bar1``, and the latch
    lives on the bar1 object. The communicator therefore snapshots the fact
    while the chain is still whole, so ``barlink_reopen()`` can refuse on the
    post-sleep shape and not only on the never-slept one.
    """
    comm = barlink_mod.BarlinkCommunicator.__new__(barlink_mod.BarlinkCommunicator)
    comm.group = "tp:0"
    comm._closed = False
    comm.transport = _MatrixStub(captured=True)

    assert comm.captured_launches() is True

    barlink_mod.BarlinkCommunicator.close(comm)

    assert comm.transport.bar1 is None, "the matrix close() erased the latch"
    assert comm.captured_launches() is True, (
        "the fact was lost with the bar1 object: a wake could then rebuild "
        "the transport under live graphs without anything noticing"
    )


def test_close_without_capture_leaves_no_capture_claim():
    """Can-fail half of the snapshot: no graphs, no claim."""
    comm = barlink_mod.BarlinkCommunicator.__new__(barlink_mod.BarlinkCommunicator)
    comm.group = "tp:0"
    comm._closed = False
    comm.transport = _MatrixStub(captured=False)

    assert comm.captured_launches() is False
    barlink_mod.BarlinkCommunicator.close(comm)
    assert comm.captured_launches() is False, (
        "a snapshot that is always True would refuse every wake"
    )


def test_reopen_refuses_while_graphs_hold_the_transport_kernels(coord):
    """A rebuild under live CUDA graphs is a use-after-free, so it is refused.

    The captured kernels reference transport-lifetime objects: ``_step_dev``
    (``barlink_bar1.py:2607``), ``_result_gen_dev`` (``:2615`` -- "it must
    keep counting on every graph replay") and the reserved graph slots of the
    result ring (``:2499-2501``, "Each captured call site takes ONE graph
    slot and does not give it back"). ``close()`` frees all of it and hands
    the BAR1 aperture back -- ``vmm_free`` = ``cuMemUnmap`` / ``cuMemRelease``
    / ``cuMemAddressFree``, so the virtual addresses go too -- and
    ``_build_barlink()`` then allocates fresh ones at fresh addresses. The
    wake sequence of the design spec (S2.5 step 3) SKIPS ``resume("cuda_graph")``
    in V1 and re-captures nothing, so the resident graphs would replay against
    freed VRAM and unmapped BAR1 pages.

    The refusal must land BEFORE the close, or it refuses a group it has
    already destroyed.
    """
    coord._build_barlink()
    comm = coord.barlink_comm
    comm.transport = _MatrixStub(captured=True)
    prior_ledger = ledger.ledger_balance(coord.device)
    prior_gate = gate.registered()

    with pytest.raises(RuntimeError, match="captured"):
        coord.barlink_reopen()

    assert coord.barlink_comm is comm, "the refusal must precede the close"
    assert not comm.closed, (
        "refusing after the close would destroy exactly what the refusal "
        "exists to protect"
    )
    assert _FakeComm.built == 1, "nothing may be rebuilt"
    assert ledger.ledger_balance(coord.device) == prior_ledger
    assert gate.registered() == prior_gate


def test_reopen_refuses_on_the_post_sleep_shape_too(coord):
    """The shape the S2 slice boot drives: sleep closed it, wake rebuilds.

    By then the bar1 object is gone (``_MatrixStub``/``BarlinkMatrixTransport``
    null it), so the refusal reads the snapshot the communicator took at
    close time -- the state ``test_close_remembers_that_graphs_held_the_
    transport_kernels`` proves the real class produces.
    """
    coord._build_barlink()
    comm = coord.barlink_comm
    comm.transport = _MatrixStub(captured=True)
    comm.transport.close()
    comm._closed_with_captured_launches = True

    with pytest.raises(RuntimeError, match="captured"):
        coord.barlink_reopen()


def test_a_reopen_without_captured_graphs_still_proceeds(coord):
    """Can-fail half of the refusal: the latch is what causes it."""
    coord._build_barlink()
    coord.barlink_comm.transport = _MatrixStub(captured=False)

    coord.barlink_reopen()

    assert _FakeComm.built == 2, "an unarmed latch must not block the wake"
