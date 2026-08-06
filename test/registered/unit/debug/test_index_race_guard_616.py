"""#616: hermetic tests for the index-race guard instrument.

The instrument is the thing the on-card window depends on, so it needs its own
can-fail proof BEFORE it is trusted to report anything: every assertion here is
written so that it goes RED if the corresponding mechanism is silently inert.
In particular ``test_mutation_between_snapshot_and_check_is_counted`` forces
exactly the corruption shape the bug shows (a SUBSET of lanes rewritten between
production and consumption) and requires the guard to name it -- if the
stability check were a no-op, that test fails rather than passing quietly.
"""

import logging

import pytest
import torch

from sglang.srt.debug_utils import index_race_guard


@pytest.fixture(autouse=True)
def _reset():
    index_race_guard._reset_for_test(enabled=True, clamp=False)
    yield
    index_race_guard._reset_for_test(enabled=False, clamp=False)


def _stats_row(name):
    st = index_race_guard._state
    slot = st.slots[name]
    st.mirror.copy_(st.stats)
    return st.mirror[slot]


def test_disabled_guard_is_identity_and_allocates_nothing():
    index_race_guard._reset_for_test(enabled=False)
    values = torch.tensor([0, 1, 2], dtype=torch.int32)
    assert index_race_guard.guard("x", values, 0, 3) is values
    index_race_guard.snapshot("x", values)
    index_race_guard.check_stable("x", values)
    index_race_guard.poll()
    # No state object at all: the disabled path must not even build one.
    assert index_race_guard._state is None


def test_in_range_values_report_no_violation():
    values = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    index_race_guard.guard("clean", values, 0, 4)
    row = _stats_row("clean")
    assert int(row[index_race_guard._F_BAD]) == 0
    assert int(row[index_race_guard._F_MIN]) == 0
    assert int(row[index_race_guard._F_MAX]) == 3


def test_out_of_range_lanes_are_counted_not_raised():
    # The crash shape: only a SUBSET of lanes out of range.
    values = torch.tensor([0, 99, 2, -7], dtype=torch.int32)
    out = index_race_guard.guard("dirty", values, -1, 4)
    row = _stats_row("dirty")
    assert int(row[index_race_guard._F_BAD]) == 2
    assert int(row[index_race_guard._F_MAX]) == 99
    assert int(row[index_race_guard._F_MIN]) == -7
    # Non-fatal and non-mutating without clamp: the caller still sees the
    # original tensor, so arming the guard cannot change behaviour by itself.
    assert out is values


def test_negative_sentinel_is_in_range_for_accept_index():
    # accept_index is initialised to -1 and -1 is a legitimate "not accepted"
    # sentinel, so the guard must not flag it. If this ever goes red the guard
    # would drown the real signal in false positives every single round.
    values = torch.full((4, 4), -1, dtype=torch.int32)
    index_race_guard.guard("sentinel", values, -1, 16)
    assert int(_stats_row("sentinel")[index_race_guard._F_BAD]) == 0


def test_clamp_forces_values_back_into_range():
    index_race_guard._reset_for_test(enabled=True, clamp=True)
    values = torch.tensor([-9, 5, 100], dtype=torch.int32)
    out = index_race_guard.guard("clamped", values, 0, 8)
    assert out.tolist() == [0, 5, 7]
    # Still reported: clamping must not suppress the evidence it acts on.
    assert int(_stats_row("clamped")[index_race_guard._F_BAD]) == 2


def test_mutation_between_snapshot_and_check_is_counted():
    """The discriminator: a tensor rewritten after production is named."""
    values = torch.arange(16, dtype=torch.int32)
    index_race_guard.snapshot("ai", values)
    # Simulate the foreign-stream write: a subset of lanes replaced, exactly
    # the 8-of-16 shape the reproduction showed.
    values[[1, 4, 5, 8, 12, 13, 14, 15]] = 123456
    index_race_guard.check_stable("ai", values)
    assert int(_stats_row("ai")[index_race_guard._F_MUT]) == 8


def test_unmutated_tensor_reports_zero_mutations():
    values = torch.arange(16, dtype=torch.int32)
    index_race_guard.snapshot("stable", values)
    index_race_guard.check_stable("stable", values)
    assert int(_stats_row("stable")[index_race_guard._F_MUT]) == 0


def test_snapshot_is_a_private_copy_not_an_alias():
    # If snapshot() aliased the guarded tensor, every later write would update
    # the reference too and check_stable could never fire -- the failure mode
    # that would make the whole instrument silently useless.
    values = torch.arange(4, dtype=torch.int32)
    index_race_guard.snapshot("alias", values)
    assert index_race_guard._state.snapshots["alias"].data_ptr() != values.data_ptr()


def test_poll_reports_each_violation_once(caplog):
    values = torch.tensor([77], dtype=torch.int32)
    with caplog.at_level(logging.ERROR):
        index_race_guard.guard("reported", values, 0, 4)
        index_race_guard.poll()
        first = [r for r in caplog.records if "INDEX-RACE" in r.getMessage()]
        assert len(first) == 1
        assert "site=reported" in first[0].getMessage()
        caplog.clear()
        # No new violation -> no repeat line (the counters are monotonic, so a
        # naive implementation would re-report the same total forever).
        index_race_guard.poll()
        assert not [r for r in caplog.records if "INDEX-RACE" in r.getMessage()]


def test_empty_tensor_is_ignored():
    empty = torch.zeros((0,), dtype=torch.int32)
    assert index_race_guard.guard("empty", empty, 0, 4) is empty
    assert index_race_guard._state is None


def test_summary_lists_every_site():
    index_race_guard.guard("a", torch.tensor([1], dtype=torch.int32), 0, 4)
    index_race_guard.guard("b", torch.tensor([9], dtype=torch.int32), 0, 4)
    index_race_guard.poll()
    text = index_race_guard.summary()
    assert "a:" in text and "b:" in text
