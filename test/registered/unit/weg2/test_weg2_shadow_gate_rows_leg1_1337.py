# SPDX-License-Identifier: Apache-2.0
"""#1337 -- LEG 1's DESTINATION CANNOT WAIT FOR ROWS THAT DO NOT EXIST YET.

Rooted by boot seat 3 on XSN12 (appended to BOOT_weg2xsn12_0911.md) and it is
an expectation defect, not a gate defect. `_weg2_shadow_gate_rows` returns
`None` (= "expect all six rows") for the DESTINATION hook, justified in its own
docstring by:

    "the destination expects all six, because the source rows were sealed
     earlier in this same flip with this region's epoch_hash and this leg"

That sentence is FALSE ON THE FIRST FLIP. On leg 1 the source rows cannot yet
exist, so group D's leg-1 destination waits the full
`weight_exchange_shadow.SHADOW_GATE_BUDGET_S` = 5.0 s for three rows nobody can
write -- measured 4.953 / 4.956 / 4.960 s -- and then reads `ran=no`. That is
6/24 legs per group on XSN12, and under `inject=authoritative` it would be
weight bytes nobody injected.

THE FIX IS THE EXPECTATION, NOT THE GATE (operator ruling, and the gate is boot
seat 3's file which this slice may not edit): on a leg whose source rows cannot
yet exist, the destination expects its OWN group's rows, exactly as the source
already does for the mirror-image reason (the source would otherwise wait on
rows that cannot be written until it stops waiting -- the circular wait the
S5b refuter found).

`leg` IS REQUIRED, with no default. A default would let a caller that forgets
it keep the old behaviour silently, which is precisely how this survived: the
call site had `leg` in scope the whole time (`weight_updater.py:2078`) and
simply did not pass it.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler_components import weight_updater as wu  # noqa: E402
from sglang.srt.weg2 import weight_exchange_region as xr  # noqa: E402
from sglang.srt.weg2 import weight_exchange_shadow as sh  # noqa: E402


def _mgr():
    return wu.SchedulerWeightUpdaterManager(
        tp_worker=None, draft_worker=None, tp_cpu_group=None,
        memory_saver_adapter=None, flush_cache=lambda *a, **k: True,
        is_fully_idle=lambda *a, **k: True,
    )


def _own(group):
    return tuple(xr.rank_row(group, r) for r in range(xr.N_CARDS))


def test_the_destination_on_leg_1_expects_its_own_rows_only():
    """THE FIX. Leg 1 has no source rows, so waiting for them is the 5 s wall.

    Measured on XSN12: 4.953/4.956/4.960 s against
    SHADOW_GATE_BUDGET_S = 5.0, then ran=no.
    """
    assert sh.SHADOW_GATE_BUDGET_S == 5.0, "the budget this defect burns"
    for group in ("P", "D"):
        got = _mgr()._weg2_shadow_gate_rows(sh.HOOK_DESTINATION, group, leg=1)
        assert got == _own(group), group
        assert len(got) == xr.N_CARDS


def test_the_destination_on_later_legs_still_expects_all_six():
    """The designed asymmetry is KEPT where its justification holds.

    From leg 2 on, the previous flip sealed the source rows, so the
    destination's cross-group expectation is real evidence and must not be
    weakened -- narrowing it everywhere would silently stop checking the
    agreement the gate exists for.
    """
    for group in ("P", "D"):
        for leg in (2, 3, 7):
            assert _mgr()._weg2_shadow_gate_rows(
                sh.HOOK_DESTINATION, group, leg=leg) is None, (group, leg)


def test_the_source_expects_its_own_rows_on_every_leg():
    """Unchanged: the S5b circular wait the asymmetry was introduced for."""
    for group in ("P", "D"):
        for leg in (1, 2, 5):
            assert _mgr()._weg2_shadow_gate_rows(
                sh.HOOK_SOURCE, group, leg=leg) == _own(group), (group, leg)


def test_an_unknown_group_still_expects_all_six():
    assert _mgr()._weg2_shadow_gate_rows(sh.HOOK_SOURCE, "?", leg=1) is None
    assert _mgr()._weg2_shadow_gate_rows(
        sh.HOOK_DESTINATION, "?", leg=1) is None


def test_leg_is_required_so_a_caller_cannot_silently_keep_the_defect():
    """The defect survived because the call site HAD `leg` and did not pass it.

    A default would restore exactly that failure mode, so the parameter is
    mandatory and a two-argument call is a TypeError rather than the old
    behaviour.
    """
    with pytest.raises(TypeError):
        _mgr()._weg2_shadow_gate_rows(sh.HOOK_DESTINATION, "D")


def test_the_call_site_passes_the_leg_it_already_had():
    """Pinned by substitution over the source, because the bug WAS the wiring.

    A correct function reached by a call site that drops the leg is still the
    5 s wall, and no unit test of the function alone can see that.
    """
    import inspect

    # NORMALISED, because the call is wrapped across two lines and a literal
    # match is brittle in exactly the way that has now bitten this campaign
    # four times: the assertion breaks on formatting while the property holds.
    src = " ".join(inspect.getsource(
        wu.SchedulerWeightUpdaterManager._weg2_shadow_hook).split())
    assert "_weg2_shadow_gate_rows(str(hook), group, leg=int(leg))" in src, src
