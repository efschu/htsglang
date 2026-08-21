# SPDX-License-Identifier: Apache-2.0
"""#785 rung 4 WIRING: the deferral gate, and the interlock that keeps it inert.

``arena_tail_probe.post_sizing_stack_bytes`` already knows how to express the credit
(``cold_stack_deferred=True``); what it lacked was anyone passing it. Passing it
is only safe when the boot ACTUALLY defers the cold posts, so the credit and the
deferral must be driven by ONE predicate rather than by two flags that can
disagree. A pool sized for a deferral that did not happen is the #678 OOM.

The predicate is the ladder depth: rung 4 ('draft+graphs') deferring the cold
stack IS the rung. Until ``IMPLEMENTED_DEPTH`` is raised -- which happens only
after a real flip cycle has executed the restore on metal -- depth 4 is refused
at resolution, so every test below that asks for the credit has to go through
the experimental hatch. That refusal IS the interlock.
"""

import sys

import pytest

from sglang.srt.managers import phase_flip_spill as sp
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10)


class _Args:
    def __init__(self, depth):
        self.phase_flip_spill_depth = depth


def _clear_env(monkeypatch):
    monkeypatch.delenv(sp.DEPTH_ENV, raising=False)
    monkeypatch.delenv(sp.DEPTH_UNIMPLEMENTED_ENV, raising=False)


# --------------------------------------------------------------------------
# The predicate
# --------------------------------------------------------------------------


def test_the_cold_stack_is_deferred_only_at_the_rung_that_names_it(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv(sp.DEPTH_UNIMPLEMENTED_ENV, "1")
    assert sp.cold_stack_deferred(_Args("draft+graphs")) is True


@pytest.mark.parametrize("depth", ["none", "cache", "draft", "arena"])
def test_no_lower_rung_defers_the_cold_stack(monkeypatch, depth):
    """CAN-FAIL GUARD. The rig serves on 'arena' today. If the deferral leaked
    down to it, the running instance would size a pool for cold posts that its
    boot still allocates -- an OOM on the next restart, not a slow path."""
    _clear_env(monkeypatch)
    assert sp.cold_stack_deferred(_Args(depth)) is False


def test_an_unconfigured_instance_never_defers(monkeypatch):
    _clear_env(monkeypatch)
    assert sp.cold_stack_deferred(None) is False
    assert sp.cold_stack_deferred(_Args(None)) is False


# --------------------------------------------------------------------------
# The interlock: rung 4 stays refused until it is exercised
# --------------------------------------------------------------------------


def test_rung_four_is_still_refused_without_the_hatch(monkeypatch):
    """The recorded refusal stands until the restore runs on metal."""
    _clear_env(monkeypatch)
    if sp.IMPLEMENTED_DEPTH >= sp.DEPTH_DRAFT_GRAPHS:
        pytest.skip("rung 4 promoted; the refusal is retired by design")
    with pytest.raises(sp.PhaseFlipSpillError, match="not wired"):
        sp.resolve_spill_depth(_Args("draft+graphs"))
    assert sp.cold_stack_deferred(_Args("draft+graphs")) is False


def test_the_hatch_opens_the_unimplemented_rung_and_says_so(monkeypatch, caplog):
    """The exercise boot needs to REACH the rung to exercise it. The hatch is
    the only way, it is loud, and it is not the default."""
    _clear_env(monkeypatch)
    monkeypatch.setenv(sp.DEPTH_UNIMPLEMENTED_ENV, "1")
    with caplog.at_level("WARNING"):
        assert sp.resolve_spill_depth(_Args("draft+graphs")) == sp.DEPTH_DRAFT_GRAPHS
    assert any("experimental" in r.message.lower() for r in caplog.records)


def test_the_hatch_does_not_invent_depths_beyond_the_ladder(monkeypatch):
    """CAN-FAIL GUARD: the hatch lifts the IMPLEMENTED bound, never MAX_DEPTH."""
    _clear_env(monkeypatch)
    monkeypatch.setenv(sp.DEPTH_UNIMPLEMENTED_ENV, "1")
    with pytest.raises(sp.PhaseFlipSpillError, match="out of range"):
        sp.resolve_spill_depth(_Args(sp.MAX_DEPTH + 1))


def test_the_deferral_predicate_never_raises_on_a_refused_depth(monkeypatch):
    """The sizer asks this predicate on EVERY boot, including boots that
    configured a depth this build refuses. It must answer 'no deferral', not
    take the boot down -- the refusal belongs to the ladder's own resolution
    path, which the scheduler reaches separately and loudly."""
    _clear_env(monkeypatch)
    if sp.IMPLEMENTED_DEPTH >= sp.DEPTH_DRAFT_GRAPHS:
        pytest.skip("rung 4 promoted; nothing is refused")
    assert sp.cold_stack_deferred(_Args("draft+graphs")) is False


# --------------------------------------------------------------------------
# The sizer call site actually passes it (part 3 of the wiring)
# --------------------------------------------------------------------------

MIB = 1048576
_RANK0_PP = int(16362.72 * MIB)
_RANK0_TP = int(15925.80 * MIB)
_RANK0_DRAFT = int(1441.14 * MIB)


class _Runner:
    """The narrowest stand-in that ``_post_sizing_stack_bytes`` actually uses."""

    def __init__(self, depth):
        self.server_args = _Args(depth)
        self._arena_tail_derivation = (_RANK0_PP, _RANK0_TP)

    def _seam_world_rank(self):
        return 0

    def _flip_draft_residency_bytes(self):
        return _RANK0_DRAFT


def _charge(depth):
    from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
        ModelRunnerKVCacheMixin,
    )

    return ModelRunnerKVCacheMixin._post_sizing_stack_bytes(_Runner(depth))


def test_the_sizer_takes_the_credit_at_the_rung_that_defers(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv(sp.DEPTH_UNIMPLEMENTED_ENV, "1")
    from sglang.srt.managers.arena_tail_probe import STACK_RESIDUAL_MIB

    assert _charge("arena") - _charge("draft+graphs") == STACK_RESIDUAL_MIB[0] * MIB


def test_the_sizer_charges_the_full_stack_at_every_lower_rung(monkeypatch):
    """CAN-FAIL GUARD, and the one that protects the RUNNING instance: the rig
    serves on 'arena'. A credit leaking to it sizes a pool for posts the boot
    still builds."""
    _clear_env(monkeypatch)
    full = _charge("arena")
    for depth in ("none", "cache", "draft", None):
        assert _charge(depth) == full


def test_the_sizer_does_not_take_the_credit_while_the_rung_is_refused(monkeypatch):
    """THE INTERLOCK, seen from the sizer. Without the hatch, asking for rung 4
    must size exactly as rung 3 does -- never credit a deferral the boot's own
    ladder is going to refuse to perform."""
    _clear_env(monkeypatch)
    if sp.IMPLEMENTED_DEPTH >= sp.DEPTH_DRAFT_GRAPHS:
        pytest.skip("rung 4 promoted; nothing is refused")
    assert _charge("draft+graphs") == _charge("arena")


def test_without_a_derivation_the_charge_is_zero_regardless_of_depth(monkeypatch):
    """Pre-existing contract: no layout derivation, no post. The deferral must
    not turn that 0 into a negative or into a credit."""
    _clear_env(monkeypatch)
    monkeypatch.setenv(sp.DEPTH_UNIMPLEMENTED_ENV, "1")
    from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
        ModelRunnerKVCacheMixin,
    )

    runner = _Runner("draft+graphs")
    runner._arena_tail_derivation = None
    assert ModelRunnerKVCacheMixin._post_sizing_stack_bytes(runner) == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
