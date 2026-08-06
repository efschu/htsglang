# SPDX-License-Identifier: Apache-2.0
"""The BAR1 abort message must be a discriminator, not a disjunction (#583).

Production crash 2026-08-06 05:53:59. Ranks 0 and 1 raised
``Bar1CollectiveAborted`` from a cuda-graph replay boundary. The message said:

    Last collective launched: all_to_all (8 bytes, 0 rounds); 0 collective(s)
    ran since the previous check, so the abort is in that window and the named
    one is its most recent member. ... Cause: either the kernel exceeded its
    cycle deadline ... or the host abort word was set.

Both halves of that were misleading, and both were decidable at the point of
the raise:

* ATTRIBUTION. ``_note_launch`` stores ``_last_op`` on every launch including
  captured ones, but does NOT advance ``_unchecked_launches`` under capture.
  With zero host-path launches in the window the named collective is
  therefore not in the window at all -- it is the last launch ever recorded,
  which at a replay boundary is a graph-capture artefact. Triage went looking
  for a decode-loop seam issuing an 8-byte ``all_to_all``. None exists.
* LABEL. ``_a2a_one_round`` recorded a bare ``"all_to_all"`` although the same
  kernel serves ``barlink_broadcast`` and ``barlink_all_gather``. The 8 bytes
  were a broadcast.
* CAUSE. The host abort word's state is local and exact: ``AbortWindow``
  records the reason it was tripped with. Offering "either deadline or host
  word" when the window exists and is untripped throws away the one fact that
  separates a peer that never arrived from a deliberate abort.

These tests are hermetic: no CUDA, no group, no transport bring-up. The
transport is built with ``object.__new__`` and given exactly the attributes
``check_aborted`` reads.
"""

from __future__ import annotations

import pytest

from sglang.srt.distributed.device_communicators.barlink_bar1 import (
    Bar1CollectiveAborted,
    BarlinkBar1Transport,
)


class _Window:
    """Stand-in for ``barlink_liveness.AbortWindow``."""

    def __init__(self, reason=None):
        self._reason = reason

    @property
    def tripped(self):
        return self._reason is not None

    @property
    def reason(self):
        return self._reason


def _transport(*, pending, captured, window, last_op="all_to_all", nbytes=8):
    """A transport whose ``check_aborted`` will reach the raise."""
    t = object.__new__(BarlinkBar1Transport)
    t.rank = 0
    t.world = 3
    t.group = "tp:0"
    t.cap_cycles = 60_000_000_000
    t._ctl_dev = object()  # only tested against None
    t._ctl_defer = True
    t._unchecked_launches = pending
    # check_aborted folds _unchecked_launches INTO _deferred_launches before
    # reading it back, so seeding both would double-count the window.
    t._deferred_launches = 0
    t._captured_launches = True
    t._boundary_checks = 0
    t._last_op = last_op
    t._last_nbytes = nbytes
    t._last_op_captured = captured
    t._abort_window = window
    t._read_status_for_check = lambda: 1  # the sticky word says "aborted"
    t._deadline_cycles = lambda: 60_000_000_000
    return t


def _message(**kw):
    t = _transport(**kw)
    with pytest.raises(Bar1CollectiveAborted) as ei:
        t.check_aborted("cuda-graph replay")
    return str(ei.value)


# -- attribution -----------------------------------------------------------


def test_empty_window_does_not_present_the_captured_op_as_a_member():
    """The exact shape of the production crash: pending == 0, op from capture."""
    msg = _message(pending=0, captured=True, window=_Window())
    assert "GRAPH-REPLAY window" in msg
    assert "is NOT named here" in msg
    assert "recorded under CUDA-graph capture" in msg
    assert "must not be read as the culprit" in msg
    # The false claim that produced the wrong-turn triage.
    assert "its most recent member" not in msg


def test_empty_window_marks_a_host_path_op_as_from_an_earlier_window():
    msg = _message(pending=0, captured=False, window=_Window())
    assert "GRAPH-REPLAY window" in msg
    assert "from an earlier, already-closed window" in msg
    assert "its most recent member" not in msg


def test_non_empty_window_still_names_the_member():
    """The case where the old wording was right must keep saying so."""
    msg = _message(pending=3, captured=False, window=_Window())
    assert "3 collective(s) ran on the host path" in msg
    assert "its most recent member" in msg
    assert "GRAPH-REPLAY window" not in msg


# -- cause -----------------------------------------------------------------


def test_untripped_window_excludes_the_host_abort_word():
    """The discriminator the 2026-08-06 triage had to reconstruct by hand."""
    msg = _message(pending=0, captured=True, window=_Window())
    assert "was NOT set, which excludes it" in msg
    assert "A peer did not arrive." in msg
    assert "SGLANG_BARLINK_BAR1_CAP_CYCLES=60000000000" in msg
    # No unresolved disjunction.
    assert "or the host abort word was set" not in msg


def test_tripped_window_names_the_host_abort_word_and_its_reason():
    msg = _message(
        pending=0, captured=True, window=_Window("peer rank 2 (pid 262164) is gone")
    )
    assert "the host abort word was set on this rank" in msg
    assert "peer rank 2 (pid 262164) is gone" in msg
    # A host-set word is NOT the "peer did not arrive" case.
    assert "A peer did not arrive." not in msg
    assert "excludes it" not in msg


def test_absent_window_says_the_deadline_is_the_only_path():
    msg = _message(pending=0, captured=True, window=None)
    assert "no device-mapped host abort word" in msg
    assert "the only path a kernel has" in msg
    assert "A peer did not arrive." not in msg


# -- op label --------------------------------------------------------------


def _label_probe(**call_kw):
    """Capture the ``op_label`` that reaches ``_a2a_one_round``."""
    seen = []
    t = object.__new__(BarlinkBar1Transport)
    t.rank = 0
    t.world = 3
    t._geo = {"a2a_slot": 1 << 20}
    t._a2a_one_round = lambda *a, **kw: seen.append(kw.get("op_label"))
    t.barlink_all_to_all_single(
        None, None, None, [8, 8, 8], [8, 8, 8], **call_kw
    )
    return seen


def test_default_label_is_unchanged():
    assert _label_probe() == ["all_to_all"]


def test_label_reaches_the_single_round_fast_path():
    """rounds=None -> n == 1: the path broadcast and all_gather actually take."""
    assert _label_probe(op_label="broadcast") == ["broadcast"]


def test_label_reaches_the_multi_round_loop():
    seen = _label_probe(op_label="all_gather", rounds=3)
    assert seen == ["all_gather"] * 3


def test_broadcast_labels_its_own_collective():
    """An 8-byte broadcast must not be recorded as an 8-byte all_to_all."""
    import torch

    seen = []
    t = object.__new__(BarlinkBar1Transport)
    t.rank = 0
    t.world = 3
    t._up = True
    t._ext = object()
    t.a2a_on = True
    t._geo = {"a2a_slot": 1 << 20}
    t.barlink_all_to_all_single = lambda *a, **kw: seen.append(kw.get("op_label"))

    tensor = torch.zeros(1, dtype=torch.int64)  # exactly 8 bytes
    t.barlink_broadcast(None, tensor, 0)

    assert seen == ["broadcast"]


def test_all_gather_labels_its_own_collective():
    import torch

    seen = []
    t = object.__new__(BarlinkBar1Transport)
    t.rank = 0
    t.world = 3
    t._up = True
    t._ext = object()
    t.a2a_on = True
    t._geo = {"a2a_slot": 1 << 20}
    t.barlink_all_to_all_single = lambda *a, **kw: seen.append(kw.get("op_label"))

    out = t.barlink_all_gather(None, torch.zeros(4, dtype=torch.int64), 0)
    assert out is not None
    assert seen and set(seen) == {"all_gather"}
