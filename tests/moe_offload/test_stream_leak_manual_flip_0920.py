# SPDX-License-Identifier: Apache-2.0
"""#1492/#1493: the per-flip driver leak, and the manual flip that must refuse.

#1492 -- WHAT GREW OUTSIDE THE TORCH ALLOCATOR. `WEG2-DC-BREAKDOWN`'s `other
(context+driver+communicator+non-torch)` post crept on every rank of boots
weg2xsn406/408 and never came back. The BAR1 lane builds a ring of CUDA
streams per TAG, per PHASE, per LANE, on EVERY flip, and destroyed none of
them -- while every other `create_stream` in the Weg-2 transport has its
matching `destroy_stream`. The arithmetic, from the boots' own line counts:

    rank   WEG2-BAR1 mapped   x ring=4   nvml_proc creep   MiB per stream
    P PP0        140            560        +436 MiB          0.78
    D TP1        110            440        +326 MiB          0.74
    D TP2         90            360        +272 MiB          0.76

#1493 -- THE MANUAL FLIP THAT SHOULD NOT HAVE STARTED. `POST /weg2/flip` with
D awake flips D->P and immediately back P->D. On weg2xsn406 the RETURN wake
had to fund D's whole kv_cache tag out of cards P had just prefilled on, and
refused it on a device OOM. The sleeper's unreleased residency was measurable
before the first flip.

Hermetic: a recording fake for the ops object, and pure functions.
"""

from __future__ import annotations

import pytest

from sglang.srt.weg2.bar1_lanes import lane_streams
from sglang.srt.weg2.front import (
    MANUAL_FLIP_RESIDENCY_SLACK_MIB,
    manual_flip_residency_refusal,
)


class _Ops:
    def __init__(self, fail_at=None, destroy_raises=False):
        self.made, self.killed = [], []
        self._fail_at = fail_at
        self._destroy_raises = destroy_raises

    def create_stream(self, device):
        if self._fail_at is not None and len(self.made) == self._fail_at:
            raise RuntimeError("no cuda here")
        self.made.append(len(self.made) + 1)
        return self.made[-1]

    def destroy_stream(self, stream):
        if self._destroy_raises:
            raise RuntimeError("driver said no")
        self.killed.append(stream)


# --- #1492: every stream the lane makes, the lane unmakes --------------------


def test_the_ring_is_destroyed_on_the_normal_path():
    ops = _Ops()
    with lane_streams(ops, 0, 4) as streams:
        assert streams == [1, 2, 3, 4]
    assert ops.killed == [1, 2, 3, 4]


def test_the_ring_is_destroyed_when_the_transfer_raises():
    """A refusal mid-tag is the case that leaked hardest: it happens on the
    unhappy path, which is the path a flip under memory pressure takes."""
    ops = _Ops()
    with pytest.raises(ValueError):
        with lane_streams(ops, 0, 4):
            raise ValueError("credit wait timed out")
    assert ops.killed == [1, 2, 3, 4]


def test_a_stream_that_could_not_be_made_is_a_zero_and_is_not_destroyed():
    """Unchanged degradation: the desk fakes carry no stream, and destroying
    a 0 handle is not a thing anyone should ask the driver to do."""
    ops = _Ops(fail_at=2)
    with lane_streams(ops, 0, 4) as streams:
        assert streams == [1, 2, 0, 0]
    assert ops.killed == [1, 2]


def test_a_destroy_that_raises_never_escapes():
    """A teardown may not turn a completed transfer into a failure."""
    ops = _Ops(destroy_raises=True)
    with lane_streams(ops, 0, 2):
        pass


def test_a_destroy_that_raises_does_not_mask_the_real_error():
    ops = _Ops(destroy_raises=True)
    with pytest.raises(ValueError, match="the real one"):
        with lane_streams(ops, 0, 2):
            raise ValueError("the real one")


def test_an_ops_without_destroy_still_works():
    """The desk fakes and any older ops object must not start raising."""

    class _Old:
        def create_stream(self, device):
            return 7

    with lane_streams(_Old(), 0, 3) as streams:
        assert streams == [7, 7, 7]


@pytest.mark.parametrize("ring", [0, 1, 4, 8])
def test_the_balance_holds_for_every_ring_size(ring):
    ops = _Ops()
    with lane_streams(ops, 0, ring) as streams:
        assert len(streams) == ring
    assert sorted(ops.killed) == sorted(ops.made)


def test_the_measured_boot_shape_balances():
    """PP0's 140 lane runs at ring=4: 560 created, 560 destroyed, 0 left."""
    ops = _Ops()
    for _ in range(140):
        with lane_streams(ops, 0, 4):
            pass
    assert len(ops.made) == 560
    assert len(ops.killed) == 560


# --- #1493: the manual flip refuses before it destroys a serving group -------


def test_the_xsn406_shape_is_refused_by_name():
    why = manual_flip_residency_refusal(
        awake="D", sleeper="P", sleeper_used_mib=3900, sleeper_dormant_mib=1500)
    assert why is not None
    assert "W115 Weg2ManualFlipRefused" in why
    assert "group P still holds 3900 MiB" in why
    assert "dormant image of 1500 MiB" in why
    assert "2400 MiB of transient residency" in why


def test_a_sleeper_at_its_dormant_image_is_not_refused():
    assert manual_flip_residency_refusal(
        awake="D", sleeper="P", sleeper_used_mib=1500, sleeper_dormant_mib=1500) is None


def test_the_slack_is_measurement_noise_not_a_reserve():
    """Exactly at the slack is allowed; one MiB past it is not."""
    base = 1500
    at = base + MANUAL_FLIP_RESIDENCY_SLACK_MIB
    assert manual_flip_residency_refusal(
        awake="D", sleeper="P", sleeper_used_mib=at, sleeper_dormant_mib=base) is None
    assert manual_flip_residency_refusal(
        awake="D", sleeper="P", sleeper_used_mib=at + 1, sleeper_dormant_mib=base) is not None


def test_a_sleeper_below_its_dormant_image_is_not_refused():
    assert manual_flip_residency_refusal(
        awake="D", sleeper="P", sleeper_used_mib=900, sleeper_dormant_mib=1500) is None


def test_only_a_flip_that_starts_from_D_turns_around():
    """From P the handler flips P->D and stops; there is no return leg to
    starve, so this guard has nothing to say."""
    assert manual_flip_residency_refusal(
        awake="P", sleeper="D", sleeper_used_mib=9999, sleeper_dormant_mib=100) is None


@pytest.mark.parametrize("used,dormant", [(None, 1500), (3900, None), (None, None)])
def test_an_absent_reading_is_never_a_refusal(used, dormant):
    assert manual_flip_residency_refusal(
        awake="D", sleeper="P", sleeper_used_mib=used, sleeper_dormant_mib=dormant) is None


def test_an_unknown_awake_group_is_never_a_refusal():
    assert manual_flip_residency_refusal(
        awake=None, sleeper="P", sleeper_used_mib=3900, sleeper_dormant_mib=1500) is None


def test_the_refusal_says_what_it_costs_and_what_it_saves():
    why = manual_flip_residency_refusal(
        awake="D", sleeper="P", sleeper_used_mib=3900, sleeper_dormant_mib=1500)
    assert "D stays awake and serving" in why
    assert "refusing a flip costs a flip, attempting it cost a group" in why
