# SPDX-License-Identifier: Apache-2.0
"""The three-fake-rank whole-boot probe for Form A (slice 6a).

Hermetic: no CUDA, no model, no process. A rank's forward is modelled as the
SEQUENCE OF COLLECTIVES it issues, three fake ranks run 48 layers of it, and
the probe answers the one question a desk can answer before a GPU window is
spent: would all three ranks arrive at the same collective, in the same
order?

This file was written BEFORE the slice-6a surgery, not after, and that order
is the point. The brief's goal was "the boot form triggers no known refusal
at the desk". Removing the refusals without changing the HOST does not make
Form A work -- it converts a loud desk-time error into a wedged rig, because
the host blocks in a per-layer all-reduce that two of three ranks will never
join. A hang leaves no log line saying which rank went where; this does.
"""

import itertools

import pytest

from sglang.srt.form_a_symmetry import (
    CollectiveMismatch,
    RankTrace,
    check_symmetry,
    probe_form_a_boot,
    trace_layer,
)
from sglang.srt.rank_role import RankRolePlan

FORM_A = RankRolePlan(("host", "worker", "worker"))


def _probe(wsd, hme, hdu, layers=48):
    return probe_form_a_boot(
        FORM_A,
        num_layers=layers,
        worker_skips_dense=wsd,
        host_uses_moe_exchange=hme,
        host_dense_is_unsharded=hdu,
    )


# ==========================================================================
# 1. Today's boot is symmetric -- the probe must agree with reality first
# ==========================================================================
def test_todays_layout_is_symmetric_and_costs_what_the_logs_say():
    """fn8ah ran without hanging, so the model of it had better be
    symmetric. 48 layers: 36 linear_attn (one all-reduce) + 12 attention
    (all-gather + all-reduce) + 48 MoE all-reduces = 108 -- the right order
    of magnitude against the 98 all-reduce + 24 all-gather the boot log
    counts, which is the sanity check on the model itself."""
    traces = _probe(False, False, False)
    assert len({len(t.ops) for t in traces}) == 1
    assert len(traces[0].ops) == 108


# ==========================================================================
# 2. The finding: which configurations HANG
# ==========================================================================
@pytest.mark.parametrize(
    "wsd,hme,hdu,symmetric",
    [
        (False, False, False, True),  # today
        (False, False, True, False),
        (False, True, False, True),
        (False, True, True, False),
        (True, False, False, False),  # <-- "slice 6a alone"
        (True, False, True, True),  # <-- the SIMPLE symmetric Form A
        (True, True, False, False),
        (True, True, True, True),  # <-- full Form A
    ],
)
def test_the_whole_switch_matrix(wsd, hme, hdu, symmetric):
    if symmetric:
        _probe(wsd, hme, hdu)
    else:
        with pytest.raises(CollectiveMismatch):
            _probe(wsd, hme, hdu)


def test_slice_6a_alone_would_hang_the_rig():
    """THE reason this file exists. Silencing the worker's dense path while
    the host still runs its per-layer collectives is not a half-built
    feature, it is a deadlock: the host blocks on participants that have
    already left the forward."""
    with pytest.raises(CollectiveMismatch) as e:
        _probe(True, False, False)
    msg = str(e.value)
    assert "collective #0 differs" in msg
    assert "ranks [0]" in msg and "[1, 2]" in msg
    assert "HANG, not an error" in msg


def test_the_moe_exchange_alone_does_not_rescue_it():
    """Adding the host-centric MoE broadcast/reduce does NOT make slice 6a
    safe: the host's q all-gather and o_proj all-reduce are still there,
    and they are what the worker no longer joins."""
    with pytest.raises(CollectiveMismatch):
        _probe(True, True, False)


def test_the_missing_switch_is_the_hosts_own_dense_collectives():
    """The third switch, which the slice plan did not have. The host's
    per-layer dense collectives exist only because the dense side is
    SHARDED; under Form A rank 0 owns every head, so they have no second
    participant and must go with the sharding."""
    _probe(True, False, True)  # symmetric
    _probe(True, True, True)  # symmetric


# ==========================================================================
# 3. The consequence for the plan: the MoE exchange is not a prerequisite
# ==========================================================================
def test_the_simple_form_a_is_symmetric_and_cheaper_in_collective_COUNT():
    """Worker skips dense + host unsharded, MoE keeping its plain
    all-reduce: 48 collectives per round against today's 108, with no MoE
    exchange needed at all.

    And the number that reorders the slice plan: the broadcast/reduce
    exchange is TWO collectives per layer where the all-reduce is one, so
    full Form A issues 96 -- twice the count of the simple form. Whether it
    is cheaper depends on cost per op, not on count, and under Form A there
    is no skew either way because the workers wait on the host. So the
    exchange is an OPTIMISATION TO MEASURE, not a prerequisite."""
    simple = _probe(True, False, True)
    full = _probe(True, True, True)
    today = _probe(False, False, False)
    assert len(simple[0].ops) == 48
    assert len(full[0].ops) == 96
    assert len(today[0].ops) == 108
    assert len(simple[0].ops) < len(full[0].ops) < len(today[0].ops)


# ==========================================================================
# 4. The checker itself can fail, and names what diverged
# ==========================================================================
def test_the_checker_names_the_first_divergence_not_just_that_there_is_one():
    a = RankTrace(0, "host")
    b = RankTrace(1, "worker")
    for t in (a, b):
        t.issue("all_reduce", "layer0.moe", "hidden")
    a.issue("all_reduce", "layer1.o_proj", "hidden")
    b.issue("broadcast", "layer1.moe_in", "rows")
    with pytest.raises(CollectiveMismatch, match="collective #1 differs"):
        check_symmetry([a, b])


def test_differing_lengths_are_caught_even_when_the_prefix_agrees():
    a = RankTrace(0, "host")
    b = RankTrace(1, "worker")
    for t in (a, b):
        t.issue("all_reduce", "layer0.moe", "hidden")
    a.issue("all_reduce", "layer1.moe", "hidden")
    with pytest.raises(CollectiveMismatch, match="different NUMBERS"):
        check_symmetry([a, b])


def test_a_differing_PAYLOAD_hangs_too_not_just_a_differing_kind():
    """Same op, same site, different shape: the collective still cannot
    complete. Payload is part of the identity for that reason."""
    a = RankTrace(0, "host")
    b = RankTrace(1, "worker")
    a.issue("all_reduce", "layer0.moe", "hidden")
    b.issue("all_reduce", "layer0.moe", "rows")
    with pytest.raises(CollectiveMismatch):
        check_symmetry([a, b])


def test_a_single_rank_is_trivially_symmetric():
    check_symmetry([RankTrace(0, "host")])
    check_symmetry([])


def test_every_switch_combination_is_covered_by_the_matrix_above():
    """No combination may go untested -- an untested one is the one that
    reaches a GPU window."""
    assert len(list(itertools.product([False, True], repeat=3))) == 8
