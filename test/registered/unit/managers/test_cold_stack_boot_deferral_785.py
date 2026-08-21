# SPDX-License-Identifier: Apache-2.0
"""#785 rung 4, part 1+2: the boot DEFERS the cold posts, the cutover BUILDS them.

The flip's TP stack is built at boot, but the PP phase runs on the boot stack
and never executes a TP or draft forward. Two of that build's posts are
therefore PHASE-COLD -- the attention-backend workspaces and the decode CUDA
graphs -- and on rank 0 of this rig they measure 2294 MiB, which is the same
number as the pool's shortfall against the 669k reference.

Deferring them to the first pp->tp cutover is spill-before-alloc read literally:
the memory is not allocated during the phase that cannot use it. This file pins
the seam between the two sites -- ONE builder, called from either -- because the
failure that matters is not "the deferral did not happen" but "it happened at
boot AND at the cutover", i.e. a second set of workspaces and a second capture.
"""

import sys

import pytest

from sglang.srt.managers import phase_flip_boot as boot
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10)

MIB = 1048576


class _Worker:
    def __init__(self):
        self.calls = []

    def init_attention_backends(self):
        self.calls.append("backends")

    def init_cuda_graphs(self):
        self.calls.append("graphs")


class _Carrier:
    def __init__(self, ok=True, n=7):
        self.ok = ok
        self._n = n
        self.checked = 0

    def contains_all_params(self):
        self.checked += 1
        return self.ok

    def param_ptrs(self):
        return tuple(range(self._n))


def _build(tp=None, draft=None, carrier=None, where="test"):
    return boot.build_cold_stack_posts(
        tp if tp is not None else _Worker(),
        draft,
        carrier,
        where=where,
    )


# --------------------------------------------------------------------------
# The builder's contract
# --------------------------------------------------------------------------


def test_the_boot_order_is_preserved_backends_on_both_then_graphs_on_both():
    """NOT four independent calls. The draft's backends must exist before the
    TARGET captures, because the capture walks the drafter it drives."""
    tp, draft = _Worker(), _Worker()
    boot.build_cold_stack_posts(tp, draft, None, where="test")
    assert tp.calls == ["backends", "graphs"]
    assert draft.calls == ["backends", "graphs"]


def test_an_instance_without_speculation_builds_only_the_target():
    tp = _Worker()
    boot.build_cold_stack_posts(tp, None, None, where="test")
    assert tp.calls == ["backends", "graphs"]


def test_building_twice_is_refused_rather_than_repeated():
    """THE FAILURE THIS FILE EXISTS FOR. If the boot builds and the cutover
    builds again, the rank carries two sets of flashinfer workspaces and two
    captures -- 2294 MiB leaked on rank 0, with no error and no symptom until
    the pool cannot back itself."""
    tp, draft = _Worker(), _Worker()
    boot.build_cold_stack_posts(tp, draft, None, where="boot")
    boot.build_cold_stack_posts(tp, draft, None, where="cutover")
    assert tp.calls == ["backends", "graphs"]
    assert draft.calls == ["backends", "graphs"]


def test_the_second_call_reports_that_it_did_nothing():
    tp = _Worker()
    assert boot.build_cold_stack_posts(tp, None, None, where="boot") is True
    assert boot.build_cold_stack_posts(tp, None, None, where="cutover") is False


# --------------------------------------------------------------------------
# The carrier pin travels WITH the capture, not with the boot
# --------------------------------------------------------------------------


def test_the_carrier_pin_is_checked_where_the_capture_happens():
    """Rung 2's pin asserts the draft params were still inside the carrier AT
    CAPTURE. Deferring the capture without moving the pin would check an
    address set that nothing had baked yet -- a green assertion about nothing."""
    tp, draft, carrier = _Worker(), _Worker(), _Carrier(ok=True)
    boot.build_cold_stack_posts(tp, draft, carrier, where="cutover")
    assert carrier.checked == 1


def test_a_moved_draft_parameter_still_takes_the_boot_down_at_the_new_site():
    """CAN-FAIL GUARD: the refusal must survive the move, or deferring the
    capture silently retires the check that prevents #656's silent corruption."""
    tp, draft, carrier = _Worker(), _Worker(), _Carrier(ok=False)
    with pytest.raises(boot.PhaseFlipBootError, match="carrier"):
        boot.build_cold_stack_posts(tp, draft, carrier, where="cutover")


def test_a_failed_pin_does_not_mark_the_stack_as_built():
    """A refusal that still latched 'built' would let a retry skip the capture
    entirely and run a drafter with no graphs."""
    tp, draft, carrier = _Worker(), _Worker(), _Carrier(ok=False)
    with pytest.raises(boot.PhaseFlipBootError):
        boot.build_cold_stack_posts(tp, draft, carrier, where="cutover")
    carrier.ok = True
    assert boot.build_cold_stack_posts(tp, draft, carrier, where="retry") is True


# --------------------------------------------------------------------------
# The boot has exactly one path to these posts
# --------------------------------------------------------------------------


def test_the_boot_reaches_the_cold_posts_only_through_the_builder():
    """STRUCTURAL, on purpose. A second direct ``init_cuda_graphs()`` left
    behind in build_phase_flip_tp_stack would defeat the deferral while every
    behavioural test above stayed green."""
    import inspect

    src = inspect.getsource(boot.build_phase_flip_tp_stack)
    assert "init_cuda_graphs()" not in src
    assert "init_attention_backends()" not in src
    assert "build_cold_stack_posts(" in src


# --------------------------------------------------------------------------
# Part 2: the cutover restores what the boot deferred
# --------------------------------------------------------------------------


class _Stacks:
    def __init__(self, tp, draft):
        self.tp_worker = tp
        self.draft_worker = draft


class _Sched:
    def __init__(self, depth):
        class _A:
            phase_flip_spill_depth = depth

        self.server_args = _A()


def _restore(depth, tp, draft, monkeypatch, carrier=None):
    from sglang.srt.managers import phase_flip_spill as sp

    monkeypatch.delenv(sp.DEPTH_ENV, raising=False)
    monkeypatch.setenv(sp.DEPTH_UNIMPLEMENTED_ENV, "1")
    monkeypatch.setattr(sp, "carrier_of", lambda w: carrier)
    return boot.restore_deferred_cold_stack(_Sched(depth), _Stacks(tp, draft))


def test_the_cutover_builds_the_posts_the_boot_left_out(monkeypatch):
    tp, draft = _Worker(), _Worker()
    assert _restore("draft+graphs", tp, draft, monkeypatch) is True
    assert tp.calls == ["backends", "graphs"]
    assert draft.calls == ["backends", "graphs"]


def test_the_cutover_does_nothing_when_the_boot_already_built(monkeypatch):
    """The steady state at every rung below 4, and at rung 4 from the SECOND
    flip on. A restore that rebuilt would re-capture on every flip -- exactly
    the trade the recorded refusal rejected, reintroduced by accident."""
    tp, draft = _Worker(), _Worker()
    boot.build_cold_stack_posts(tp, draft, None, where="boot")
    assert _restore("draft+graphs", tp, draft, monkeypatch) is False
    assert tp.calls == ["backends", "graphs"]


def test_a_non_deferring_rung_never_touches_the_workers_at_the_seam(monkeypatch):
    """CAN-FAIL GUARD for the RUNNING instance, which serves on 'arena'."""
    tp, draft = _Worker(), _Worker()
    assert _restore("arena", tp, draft, monkeypatch) is False
    assert tp.calls == []
    assert draft.calls == []


def test_the_restore_checks_the_carrier_pin_against_the_deferred_capture(
    monkeypatch,
):
    tp, draft, carrier = _Worker(), _Worker(), _Carrier(ok=True)
    assert _restore("draft+graphs", tp, draft, monkeypatch, carrier=carrier) is True
    assert carrier.checked == 1


def test_the_cutover_reaches_the_restore_through_the_named_helper():
    """STRUCTURAL: the seam's restore slot must actually call it."""
    import inspect

    from sglang.srt.managers import phase_flip_runtime as rt

    src = inspect.getsource(rt.build_production_flip_cutover)
    assert "restore_deferred_cold_stack" in src


# --------------------------------------------------------------------------
# Part 4: the pp->tp seam PRICES the restore it is about to perform
# --------------------------------------------------------------------------


def _runtime(scheduler, rank=0):
    from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

    rt = object.__new__(PhaseFlipRuntime)
    rt._census_scheduler = scheduler
    rt._rank = rank
    return rt


def _price(depth, monkeypatch, direction=None, rank=0, built=False, draft=True):
    from sglang.srt.layers.dcp.phase_flip_plan import PP_TO_TP
    from sglang.srt.managers import phase_flip_spill as sp
    from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

    monkeypatch.delenv(sp.DEPTH_ENV, raising=False)
    monkeypatch.setenv(sp.DEPTH_UNIMPLEMENTED_ENV, "1")
    tp = _Worker()
    if built:
        setattr(tp, boot.COLD_STACK_BUILT_ATTR, True)
    sched = _Sched(depth)
    sched.phase_flip_stacks = _Stacks(tp, _Worker() if draft else None)
    rt = _runtime(sched, rank)
    return PhaseFlipRuntime._cold_stack_restore_bytes(
        rt, direction if direction is not None else PP_TO_TP
    )


def test_the_seam_prices_the_deferred_build_on_the_leg_that_performs_it(monkeypatch):
    """THE WHOLE POINT OF PART 4. The build happens inside _cutover, past the
    point of no return -- the same shape that killed all three ranks on
    2026-08-09 when rung 2's re-commit was unpriced. Pricing it converts a
    death into a free abandon before a byte moves."""
    from sglang.srt.managers.arena_tail_probe import STACK_RESIDUAL_MIB

    for rank in (0, 1, 2):
        assert _price("draft+graphs", monkeypatch, rank=rank) == (
            STACK_RESIDUAL_MIB[rank] * MIB
        )


def test_nothing_is_priced_on_the_tp_to_pp_leg(monkeypatch):
    from sglang.srt.layers.dcp.phase_flip_plan import TP_TO_PP

    assert _price("draft+graphs", monkeypatch, direction=TP_TO_PP) == 0


def test_nothing_is_priced_once_the_posts_exist(monkeypatch):
    """From the second pp->tp leg on there is nothing to build, so charging
    for it would abandon flips that fit -- against a record of 0 abandons."""
    assert _price("draft+graphs", monkeypatch, built=True) == 0


def test_nothing_is_priced_at_a_rung_that_does_not_defer(monkeypatch):
    """CAN-FAIL GUARD for the running instance on 'arena': a phantom charge
    would shrink its staging budget and abandon flips for a build that never
    happens."""
    for depth in ("none", "cache", "draft", "arena"):
        assert _price(depth, monkeypatch) == 0


def test_a_unit_stub_without_a_scheduler_prices_zero_rather_than_raising():
    from sglang.srt.layers.dcp.phase_flip_plan import PP_TO_TP
    from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

    rt = _runtime(None)
    assert PhaseFlipRuntime._cold_stack_restore_bytes(rt, PP_TO_TP) == 0


def test_the_staging_formula_actually_consults_the_new_term():
    """STRUCTURAL: a priced term nobody adds is not a gate."""
    import inspect

    from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

    src = inspect.getsource(PhaseFlipRuntime._staging_bytes)
    assert "_cold_stack_restore_bytes" in src


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
