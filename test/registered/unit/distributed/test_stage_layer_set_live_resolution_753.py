"""The set-aware ownership fix never fired on metal: the lookup link was dead.

THE SPECIMEN. Gapped boot v7, ``boot_735_v7gapped.log``, 2026-08-18 17:30Z.
The tree carried BOTH halves of the #753 fix -- the configurator logged
``cell_size=0`` for PP0, and the pool call site resolved ownership through
``stage_owned_layer_ids(...)`` -- and PP0's arena STILL reserved 22.6 GiB
(15 span layers) and died in ``cuMemCreate`` at 49s.

The chain: ``stage_owned_layer_ids`` defers to ``current_stage_layer_set()``,
which read ``getattr(get_pp_group(), "num_hidden_layers", None)`` -- an
attribute NO site in the tree ever stamps onto the group object.  So on every
live process it returned None, ``stage_owned_layer_ids`` degraded to the
interval test, and the set-aware branch was structurally unreachable at the
pool call site.  The desk tests all passed ``owned=`` explicitly and never
walked the lookup.

THE FIX pinned here: ``current_stage_layer_set()`` no longer needs the
attribute.  When ``SGLANG_PP_LAYER_SET`` is present, the layer count is
recovered from the set string itself -- ``parse_pp_layer_sets`` enforces full
cover of ``[0, N)``, so ``N == max(layer) + 1`` exactly.

Hermetic: no CUDA, no process group -- the pp group is a stub WITHOUT
``num_hidden_layers``, which is precisely the live condition.
"""

import os
import unittest
from unittest import mock

import sglang.srt.distributed.utils as dist_utils
from sglang.srt.distributed.utils import (
    PP_CROSSING_WIRE_ENV,
    PP_LAYER_SET_ENV,
    current_stage_layer_set,
    stage_owned_layer_ids,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# The v7 boot's exact layer set: 48 GDN on PP0, 8 FA on PP1, 8 FA on PP2.
GAPPED_SET = (
    "0-2,4-6,8-10,12-14,16-18,20-22,24-26,28-30,32-34,36-38,40-42,44-46,"
    "48-50,52-54,56-58,60-62;3,7,11,15,19,23,27,31;35,39,43,47,51,55,59,63"
)
FULL_ATTENTION_LAYER_IDS = list(range(3, 64, 4))
PP0_OWNED = frozenset(i for i in range(0, 63) if i % 4 != 3)
PP1_OWNED = frozenset(range(3, 32, 4))
PP2_OWNED = frozenset(range(35, 64, 4))


class _GroupWithoutTheAttribute:
    """The live pp group as it actually is: rank and size, nothing stamped."""

    def __init__(self, rank_in_group, world_size):
        self.rank_in_group = rank_in_group
        self.world_size = world_size


class _StageEnv:
    def __init__(self, rank, world=3, raw=GAPPED_SET, wire="1"):
        self._env = {PP_LAYER_SET_ENV: raw, PP_CROSSING_WIRE_ENV: wire}
        self._group = _GroupWithoutTheAttribute(rank, world)

    def __enter__(self):
        self._patches = [
            mock.patch.dict(os.environ, self._env),
            mock.patch(
                "sglang.srt.distributed.get_pp_group", return_value=self._group
            ),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()
        return False


class TestTheLookupResolvesWithoutTheStamp(CustomTestCase):
    """Red on the pre-fix tree: every one of these returned None / the span."""

    def test_pp0_resolves_its_gapped_set(self):
        with _StageEnv(rank=0):
            self.assertEqual(current_stage_layer_set(), PP0_OWNED)

    def test_pp1_and_pp2_resolve_theirs(self):
        for rank, want in ((1, PP1_OWNED), (2, PP2_OWNED)):
            with self.subTest(rank=rank), _StageEnv(rank=rank):
                self.assertEqual(current_stage_layer_set(), want)

    def test_the_pool_call_site_now_gets_zero_layers_on_pp0(self):
        """The exact call the mixin makes -- no explicit ``owned=``."""
        with _StageEnv(rank=0):
            got = stage_owned_layer_ids(FULL_ATTENTION_LAYER_IDS, 0, 63)
        self.assertEqual(got, [])

    def test_the_span_would_have_claimed_fifteen(self):
        """The refuted resolution, kept as the magnitude of the defect."""
        span = [i for i in FULL_ATTENTION_LAYER_IDS if 0 <= i < 63]
        self.assertEqual(len(span), 15)

    def test_pp1_pool_call_site_is_set_aware_too(self):
        with _StageEnv(rank=1):
            got = stage_owned_layer_ids(FULL_ATTENTION_LAYER_IDS, 3, 32)
        self.assertEqual(got, sorted(PP1_OWNED & set(FULL_ATTENTION_LAYER_IDS)))


class TestTheContiguousPathIsUntouched(CustomTestCase):
    def test_no_env_means_none_without_touching_the_group(self):
        env = {k: v for k, v in os.environ.items() if k != PP_LAYER_SET_ENV}
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch(
                "sglang.srt.distributed.get_pp_group",
                side_effect=AssertionError("group must not be consulted"),
            ):
                self.assertIsNone(current_stage_layer_set())

    def test_flip_world_of_one_still_answers_none(self):
        """#754: the TP stack re-reads the env with pp world size 1."""
        with _StageEnv(rank=0, world=1):
            self.assertIsNone(current_stage_layer_set())

    def test_a_stamped_group_still_wins(self):
        with _StageEnv(rank=0):
            group = _GroupWithoutTheAttribute(0, 3)
            group.num_hidden_layers = 64
            with mock.patch(
                "sglang.srt.distributed.get_pp_group", return_value=group
            ):
                self.assertEqual(current_stage_layer_set(), PP0_OWNED)


class TestTheDerivedLayerCount(CustomTestCase):
    def test_the_gapped_set_derives_sixty_four(self):
        self.assertEqual(
            dist_utils._num_layers_from_layer_set_raw(GAPPED_SET), 64
        )

    def test_singletons_and_ranges_mix(self):
        self.assertEqual(
            dist_utils._num_layers_from_layer_set_raw("0-3;4,6,5;7"), 8
        )

    def test_garbage_degrades_to_none_not_a_raise(self):
        self.assertIsNone(dist_utils._num_layers_from_layer_set_raw("a-b;c"))
        self.assertIsNone(dist_utils._num_layers_from_layer_set_raw(";;"))


if __name__ == "__main__":
    unittest.main()
