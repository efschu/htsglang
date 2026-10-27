# SPDX-License-Identifier: Apache-2.0
"""Dual W4A16 (variant 'dualm', ``marlin_native_inplace`` on the sm_86 P ranks): P's sleep/wake
must leave the Marlin content of the shared part alone (desk order F, 27.10.).

WHY THIS IS PINNED. The two flip hooks of ``weight_updater``
(``_pdflip_nvfp4_marlin_to_native`` at the first weights pause of a sleep,
``_pdflip_nvfp4_marlin_after_wake`` after the resume) walk ``_pdflip_wake_models()`` with
``nvfp4_marlin_inplace.flagged_layers``. On a dual P rank that list is the stage HULL, and the hull's
linears are shells (``dual_group_lane.Lane*Shell``) that keep their N parts in a plain TUPLE
(``_lane_parts``), deliberately not registered as submodules ("the shared parts are the serving
group's resident modules and must not be double-walked", dual_group_lane.py ~l.226). So
``hull.modules()`` never reaches the part layers that carry ``_nvfp4_marlin_inplace``: the hooks are
inert on the dual P sleep, and that is CORRECT -- the local part's bytes are D's live bytes (the union
image) and D keeps decoding on them while P sleeps; converting them to native at P's sleep would
make D read native bytes as Marlin. The non-local parts' Marlin bytes round-trip through the weights
CPU backup unchanged.

The inertness is structural and accidental. This file turns it into a pinned fact: if somebody
registers the parts as submodules (or passes the part models to the hooks), these tests fail and the
change must first answer "who converts D's bytes while D is running" (see
deskq/dual-w4a16-risiken-1027.md, risk R2).

CPU, hermetic.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from unittest import mock

import torch

from flliper.srt.layers.quantization import nvfp4_marlin_inplace as mi
from flliper.srt.model_executor.dual_group_lane import LaneColumnParallelShell
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

N, K = 128, 128  # one scale tile, K % 64 == 0: the smallest layer band_table accepts


def _marlin_part(seed: int) -> torch.nn.Module:
    """A flagged, Marlin-stamped part linear as ``prepare_layer`` leaves it, plus the attributes a
    column-parallel shell reads from its parts."""
    g = torch.Generator().manual_seed(seed)
    part = torch.nn.Module()
    part.weight = torch.nn.Parameter(
        torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, generator=g), requires_grad=False
    )
    part.weight_scale = torch.nn.Parameter(
        torch.randint(0, 0x7F, (N, K // 16), dtype=torch.uint8, generator=g).view(torch.float8_e4m3fn),
        requires_grad=False,
    )
    setattr(part, mi.BANDS_ATTR, mi.band_table(N, K))
    setattr(part, mi.LAYER_FLAG, True)
    # random bytes are NATIVE bytes; only a real conversion yields bytes that are a valid Marlin image
    mi._stamp(part.weight, mi.NATIVE)
    mi._stamp(part.weight_scale, mi.NATIVE)
    mi.layer_to_marlin(part)
    part.output_partition_sizes = (N,)
    part.gather_output = False
    part.skip_bias_add = False
    part.bias = None
    return part


def _hull_over(parts) -> torch.nn.Module:
    hull = torch.nn.Module()
    hull.qkv_shell = LaneColumnParallelShell(parts)  # keeps the parts in a plain tuple
    return hull


def _snapshot(part):
    return (part.weight.detach().clone(), part.weight_scale.detach().view(torch.uint8).clone())


def _same(part, snap) -> bool:
    return torch.equal(part.weight, snap[0]) and torch.equal(part.weight_scale.view(torch.uint8), snap[1])


def _stub(models, carrier):
    from flliper.srt.managers.scheduler_components.weight_updater import (
        SchedulerWeightUpdaterManager as WU,
    )

    stub = mock.Mock()
    stub._pdflip_wake_models = lambda: models
    stub._pdflip_wake_weight_carrier = lambda: carrier
    stub._pdflip_model_for_group = lambda g: None
    stub._pdflip_nvfp4_draft_disk_reloaded = False
    stub.CARRIER_EXCHANGE = WU.CARRIER_EXCHANGE
    return WU, stub


class TestDualPSleepHooksAreInert(CustomTestCase):
    def test_the_parts_of_a_shell_are_not_reachable_through_the_hull(self):
        parts = [_marlin_part(1), _marlin_part(2), _marlin_part(3)]
        hull = _hull_over(parts)
        self.assertEqual(mi.flagged_layers([hull]), [])
        # control: the same layers ARE found when registered in the tree
        tree = torch.nn.Module()
        for i, p in enumerate(parts):
            tree.add_module(f"p{i}", p)
        self.assertEqual({id(m) for m in mi.flagged_layers([tree])}, {id(p) for p in parts})

    def test_sleep_then_tms_backup_wake_leaves_every_part_byte_identical(self):
        parts = [_marlin_part(1), _marlin_part(2), _marlin_part(3)]
        snaps = [_snapshot(p) for p in parts]
        hull = _hull_over(parts)
        WU, stub = _stub([hull], carrier="tms-backup")
        for _ in range(3):  # three P sleeps under D pressure
            WU._pdflip_nvfp4_marlin_to_native(stub)
            WU._pdflip_nvfp4_marlin_after_wake(stub)
        for p, s in zip(parts, snaps):
            self.assertTrue(_same(p, s))
            self.assertEqual(mi._content(p.weight), mi.MARLIN)
            self.assertEqual(mi._content(p.weight_scale), mi.MARLIN)

    def test_control_a_registered_layer_would_be_permuted_at_the_same_hooks(self):
        """Why the inertness matters: were the part reachable, the sleep hook would rewrite its bytes
        in place -- for the local part those bytes are D's live image."""
        part = _marlin_part(4)
        before = _snapshot(part)
        WU, stub = _stub([part], carrier="tms-backup")
        WU._pdflip_nvfp4_marlin_to_native(stub)
        self.assertFalse(_same(part, before))
        self.assertEqual(mi._content(part.weight), mi.NATIVE)
        WU._pdflip_nvfp4_marlin_after_wake(stub)  # backup restore: the stamp is the truth -> back to Marlin
        self.assertTrue(_same(part, before))


if __name__ == "__main__":
    import unittest

    unittest.main()
