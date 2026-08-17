"""#631 x #364: the resident-slot cap must survive into the flip's SECOND stack.

MEASURED FAILURE this pins (2026-08-09, metal, boot refused):

    --gdn-resident-state-slots 10 --max-mamba-cache-size 20
    -> PhaseFlipBootError: req_to_token shapes diverge between stacks:
       PP (5, 393224) vs TP (4, 393224)

The mechanism, and it is entirely about WHERE the pre-cap count is kept.
Applying the cap OVERRIDES ``server_args.max_mamba_cache_size`` to the
capped value, and slice 3 remembered the pre-cap count on the MODEL RUNNER
(``self._gdn_profiled_state_slots``). A phase-flip instance then builds its
TP stack from ``copy.deepcopy(server_args)`` taken AFTER that override
(``phase_flip_boot``), so that stack:

  * has no runner attribute (different runner object), and
  * reads an already-capped ``max_mamba_cache_size``,

and therefore sizes its concurrency ceiling from the SHRUNKEN pool -- the
exact crater slice 3 exists to prevent, reintroduced through the copy. The
two stacks disagree on ``max_num_reqs``, their ``req_to_token`` rows differ,
and the flip's boot guard refuses the instance outright.

The fix moves the memory onto the ARGS, write-once, so the deepcopy carries
it. These cases are the can-fail set: remove the sticky read and
``test_the_two_stacks_diverge_without_the_sticky`` documents exactly what
comes back.

Both flags compose only with this in place, which matters beyond tidiness:
the cap is what buys the KV pool that lets a single session exceed the
model's native context ceiling on this rig.
"""

import copy
import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.gdn_slot_ladder import (
    PROFILED_SLOTS_ATTR,
    cap_is_binding,
    effective_state_slots,
    recall_profiled_state_slots,
    remember_profiled_state_slots,
)
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

# The metal numbers, so the pin fails with the shape it was written for.
PROFILED = 20
RESIDENT_CAP = 10
RATIO = 3
USER_MAX_RUNNING = 4


class _ArgsStub(SimpleNamespace):
    """A ServerArgs stand-in that carries the one method under test.

    ``remember_profiled_state_slots`` writes through ``ServerArgs.override``
    now, so a stub without it would fail for the wrong reason -- and a stub
    that silently swallowed the call would let the production path rot. This
    mirrors override's post-state (set the field) and records the source, so
    the tests can assert the write was audited rather than bare.
    """

    def override(self, source, **fields):
        self.overrides = getattr(self, "overrides", []) + [(source, dict(fields))]
        for name, value in fields.items():
            setattr(self, name, value)


def _args(max_mamba_cache_size, cap):
    return _ArgsStub(
        max_running_requests=USER_MAX_RUNNING,
        max_mamba_cache_size=max_mamba_cache_size,
        gdn_resident_state_slots=cap,
        dp_size=1,
    )


def _apply_cap(server_args):
    """What handle_max_mamba_cache does, reduced to the two lines that matter."""
    profiled = remember_profiled_state_slots(server_args)
    if cap_is_binding(server_args.gdn_resident_state_slots, profiled):
        server_args.max_mamba_cache_size = effective_state_slots(
            profiled, server_args.gdn_resident_state_slots
        )
    return profiled


def _runner(server_args, *, runner_attr):
    r = SimpleNamespace()
    r.server_args = server_args
    r.dp_size = 1
    r.model_config = SimpleNamespace(context_len=393216)
    r.mambaish_config = object()
    r._calculate_mamba_ratio = lambda: RATIO
    if runner_attr is not None:
        r._gdn_profiled_state_slots = runner_attr
    return r


def _resolve(r, token_capacity=277468):
    return ModelRunnerKVCacheMixin._resolve_max_num_reqs(
        r, token_capacity=token_capacity
    )


class TestProfiledSlotsMemory(CustomTestCase):
    """The write-once memory itself."""

    def test_the_attribute_is_public_so_the_guard_can_see_it(self):
        # THIS PIN IS THE INVERSE OF THE ONE IT REPLACES. The attribute used
        # to be underscore-prefixed precisely BECAUSE ServerArgs.__setattr__
        # exempts private names from the strict post-resolution guard
        # (server_args.py:17437), and the old test pinned that as the
        # mechanism the design "rests on". It was guard evasion: a real
        # post-resolution mutation made invisible to the mechanism built to
        # see it. The name is public now and the write goes through
        # override(), so the route is closed rather than unused.
        self.assertFalse(PROFILED_SLOTS_ATTR.startswith("_"))

    def test_the_write_is_audited_not_bare(self):
        a = _args(PROFILED, RESIDENT_CAP)
        remember_profiled_state_slots(a)
        sources = [src for src, _ in getattr(a, "overrides", [])]
        self.assertEqual(sources, ["gdn_slot_ladder.remember_profiled_state_slots"])

    def test_write_once_still_holds_through_override(self):
        """One override call, not one per caller."""
        a = _args(PROFILED, RESIDENT_CAP)
        remember_profiled_state_slots(a)
        remember_profiled_state_slots(a)
        remember_profiled_state_slots(a)
        self.assertEqual(len(getattr(a, "overrides", [])), 1)

    def test_first_call_records_the_pre_cap_count(self):
        a = _args(PROFILED, RESIDENT_CAP)
        self.assertEqual(remember_profiled_state_slots(a), PROFILED)
        self.assertEqual(getattr(a, PROFILED_SLOTS_ATTR), PROFILED)

    def test_it_is_write_once_across_the_override(self):
        a = _args(PROFILED, RESIDENT_CAP)
        _apply_cap(a)
        self.assertEqual(a.max_mamba_cache_size, RESIDENT_CAP)  # pool capped
        # A second application (the TP stack) must still see the PRE-cap 20.
        self.assertEqual(remember_profiled_state_slots(a), PROFILED)

    def test_the_deepcopy_carries_it(self):
        a = _args(PROFILED, RESIDENT_CAP)
        _apply_cap(a)
        tp_args = copy.deepcopy(a)  # phase_flip_boot's copy, taken post-cap
        self.assertEqual(tp_args.max_mamba_cache_size, RESIDENT_CAP)
        self.assertEqual(recall_profiled_state_slots(tp_args), PROFILED)

    def test_recall_never_writes(self):
        a = _args(PROFILED, None)
        self.assertIsNone(recall_profiled_state_slots(a))
        self.assertFalse(hasattr(a, PROFILED_SLOTS_ATTR))


class TestTwoStacksAgree(CustomTestCase):
    """The invariant the flip's boot guard checks."""

    def _two_stacks(self):
        pp_args = _args(PROFILED, RESIDENT_CAP)
        profiled = _apply_cap(pp_args)
        pp = _runner(pp_args, runner_attr=profiled)
        # The TP stack: a deepcopy taken after the cap, its own runner.
        tp_args = copy.deepcopy(pp_args)
        _apply_cap(tp_args)  # the second stack sizes its own pool too
        tp = _runner(tp_args, runner_attr=None)
        return pp, tp

    def test_both_stacks_resolve_the_same_max_num_reqs(self):
        pp, tp = self._two_stacks()
        self.assertEqual(_resolve(pp), _resolve(tp))

    def test_the_shared_value_is_the_user_ceiling(self):
        # 20 profiled // ratio 3 = 6 sessions of budget, clamped by the
        # user's --max-running-requests 4. Both stacks must land on 4, which
        # is what makes req_to_token 5 rows on both sides.
        pp, tp = self._two_stacks()
        self.assertEqual(_resolve(pp), USER_MAX_RUNNING)
        self.assertEqual(_resolve(tp), USER_MAX_RUNNING)

    def test_the_two_stacks_diverge_without_the_sticky(self):
        # CAN-FAIL: this is the old behaviour, reproduced exactly. The TP
        # stack has neither the runner attribute nor a remembered count, so
        # it falls back to the capped pool: 10 // 3 = 3 against the PP
        # stack's 4. That one-row difference is the refused boot.
        pp_args = _args(PROFILED, RESIDENT_CAP)
        profiled = _apply_cap(pp_args)
        pp = _runner(pp_args, runner_attr=profiled)
        tp_args = copy.deepcopy(pp_args)
        delattr(tp_args, PROFILED_SLOTS_ATTR)  # the memory that used to be lost
        tp = _runner(tp_args, runner_attr=None)
        self.assertEqual(_resolve(pp), 4)
        self.assertEqual(_resolve(tp), 3)
        self.assertNotEqual(_resolve(pp), _resolve(tp))

    def test_cap_unset_is_unchanged_on_both_stacks(self):
        pp_args = _args(PROFILED, None)
        pp = _runner(pp_args, runner_attr=None)
        tp_args = copy.deepcopy(pp_args)
        tp = _runner(tp_args, runner_attr=None)
        # 20 // 3 = 6, clamped by the user's 4, and nothing was recorded.
        self.assertEqual(_resolve(pp), USER_MAX_RUNNING)
        self.assertEqual(_resolve(tp), USER_MAX_RUNNING)
        self.assertFalse(hasattr(pp_args, PROFILED_SLOTS_ATTR))


if __name__ == "__main__":
    unittest.main()
