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
    same construction, hence re-registers and re-credits, and that a closed
    transport refuses rather than returning a wrong answer.

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
    """

    built = 0

    def __init__(self, cpu_group, device, group):
        assert cpu_group is not None, "a barlink build without a cpu_group"
        self.cpu_group = cpu_group
        self.device = device
        self.group = group
        self.closed = False
        self.state = {"direct": True, "achieved": "bar1"}
        type(self).built += 1
        ledger.ledger_credit(device, group, _FAKE_REGION_BYTES)
        gate.register(self)

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
    gate.reset_for_test()

    c = ps.GroupCoordinator.__new__(ps.GroupCoordinator)
    c.world_size = 3
    c.cpu_group = object()
    c.device = torch.device("cuda:0")
    c.unique_name = "tp:0"
    c.barlink_comm = None
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
    assert ledger.ledger_balance(coord.device) == prior_ledger, (
        "two live BAR1 credits for one group: the next window_for would be "
        "priced against space this process has not given back"
    )
    assert gate.registered() == [coord.barlink_comm], (
        "the abort gate must hold exactly one transport per group"
    )


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
