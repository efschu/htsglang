# SPDX-License-Identifier: Apache-2.0
"""BOOTZEIT 5 (29.09.): the model load runs with the pre-load objects frozen.

Metal (z30w-park): the presplit's "repack" posten is a FIXED cost per MoE
layer -- 0.43 s on PP0 (512 of 512 rows repacked), 0.49 s on D TP0 (29 of 201
rows), Marlin itself 0.01 s -- and the fixed piece in it is the per-layer full
``gc.collect()`` over the whole process heap (0.26-0.29 s measured on the
import graph alone, 801522 objects; 1.5 ms after ``gc.freeze()``).

What the freeze must never change: objects the load creates stay collectible,
cycles included; nothing that is already garbage gets frozen; whatever waited
is freed (and named) at the load's end; the permanent generation leaves the
load as it came in; and a load without the expert presplit -- the 27B, dense
-- is not touched at all (27B review 29.09.).
"""

import ast
import gc
import inspect
import os
import re
import tempfile
import textwrap
import unittest
import weakref
from unittest import mock

import torch

from sglang.srt.environ import PresplitGcMode, envs
from sglang.srt.layers.moe import expert_offload as eo
from sglang.srt.model_loader import load_gc
from sglang.srt.model_loader import loader as loader_mod
from sglang.srt.model_loader.load_gc import load_gc_frozen


class _Node:
    pass


def _garbage_cycle(payload):
    """A reference cycle only a collection can free; returns a weakref to it."""
    a, b = _Node(), _Node()
    a.other, b.other = b, a
    a.payload = payload
    return weakref.ref(a)


class _FreezeCase(unittest.TestCase):
    def setUp(self):
        gc.unfreeze()  # a clean permanent generation for every case
        gc.disable()  # only the explicit collections run (deterministic)
        self._clock = eo.expert_store_clock()

    def tearDown(self):
        gc.enable()
        gc.unfreeze()
        eo._STORE_CLOCK.update(self._clock)


class TestDenseLoadIsUntouched(_FreezeCase):
    """27B review (a): a load without the expert presplit freezes nothing."""

    def test_no_presplit_means_no_freeze_no_collect_no_sampler(self):
        with envs.SGLANG_OPT_LOAD_GC_FREEZE.override(True), \
                mock.patch.object(load_gc.gc, "collect") as collect, \
                mock.patch.object(load_gc.gc, "freeze") as freeze, \
                mock.patch.object(load_gc, "_PeakSampler") as sampler:
            with load_gc_frozen(what="dense", expert_presplit=False) as frozen:
                self.assertIsNone(frozen)
        collect.assert_not_called()
        freeze.assert_not_called()
        sampler.assert_not_called()
        self.assertLess(gc.get_freeze_count(), load_gc._FOREIGN_FREEZE_MIN)

    def test_default_fraction_is_no_presplit(self):
        # The 27B profiles set no resident fraction: the default 1.0 is "no
        # expert offload anywhere", so the loader passes expert_presplit=False.
        with envs.SGLANG_MOE_RESIDENT_EXPERT_FRACTION.override("1.0"):
            self.assertFalse(load_gc.expert_presplit_runs())
        with envs.SGLANG_MOE_RESIDENT_EXPERT_FRACTION.override("0.4"):
            self.assertTrue(load_gc.expert_presplit_runs())

    def test_unreadable_offload_state_does_not_freeze(self):
        with mock.patch(
            "sglang.srt.layers.moe.resident_fraction.offload_active",
            side_effect=RuntimeError("no context"),
        ):
            self.assertFalse(load_gc.expert_presplit_runs())


class TestFreezeSemantics(_FreezeCase):
    def test_cycles_the_load_creates_are_still_collected(self):
        # The [E] stacks' case: born inside, found by the per-layer collect.
        with envs.SGLANG_OPT_LOAD_GC_FREEZE.override(True):
            with load_gc_frozen(what="test", expert_presplit=True) as frozen:
                self.assertGreater(frozen, 0)
                ref = _garbage_cycle(bytearray(1 << 20))
                self.assertIsNotNone(ref())
                self.assertGreaterEqual(gc.collect(), 2)
                self.assertIsNone(ref())
        self.assertLess(gc.get_freeze_count(), load_gc._FOREIGN_FREEZE_MIN)

    def test_a_collection_before_the_load_does_not_disarm_the_freeze(self):
        # Regression (c5d2599676, caught before a boot): CPython 3.12 parks
        # immortal objects in the permanent generation on EVERY collection
        # (375 in a bare interpreter), so "freeze count > 0 = someone else
        # froze" skipped the freeze in every real process.
        gc.collect()
        self.assertGreater(gc.get_freeze_count(), 0)
        with envs.SGLANG_OPT_LOAD_GC_FREEZE.override(True):
            with load_gc_frozen(what="test", expert_presplit=True) as frozen:
                self.assertIsNotNone(frozen)
                self.assertGreater(frozen, load_gc._FOREIGN_FREEZE_MIN)

    def test_garbage_from_before_the_load_is_not_frozen(self):
        # The pre-collect: what is already garbage at the freeze is freed then.
        ref = _garbage_cycle(bytearray(1 << 20))
        with envs.SGLANG_OPT_LOAD_GC_FREEZE.override(True):
            with load_gc_frozen(what="test", expert_presplit=True):
                self.assertIsNone(ref())

    def test_what_waited_is_freed_and_named_at_the_load_end(self):
        # The bound on "held back": a live pre-load cycle holding a tensor
        # that becomes garbage DURING the load waits (frozen, not walked) and
        # is freed at the end -- and the end line names its bytes.
        holder = {}
        holder["cycle"] = _Node()
        holder["cycle"].me = holder["cycle"]
        holder["cycle"].t = torch.empty(3 * 2**20, dtype=torch.uint8)
        ref = weakref.ref(holder["cycle"])
        with envs.SGLANG_OPT_LOAD_GC_FREEZE.override(True), \
                self.assertLogs(load_gc.logger, level="INFO") as logs:
            with load_gc_frozen(what="test", expert_presplit=True):
                holder.clear()  # unreachable now, but frozen
                gc.collect()
                self.assertIsNotNone(ref())
        self.assertIsNone(ref())
        end = [m for m in logs.output if "BOOTZEIT5 LOAD-GC-FREEZE end" in m]
        self.assertEqual(len(end), 1)
        host = float(re.search(r"host=([\d.]+) MiB", end[0]).group(1))
        self.assertGreaterEqual(host, 3.0)

    def test_end_line_carries_the_reclaim_cost_per_layer(self):
        with envs.SGLANG_OPT_LOAD_GC_FREEZE.override(True), \
                envs.SGLANG_OPT_LOAD_PRESPLIT_GC.override(PresplitGcMode.FULL), \
                self.assertLogs(load_gc.logger, level="INFO") as logs:
            with load_gc_frozen(what="test", expert_presplit=True):
                eo.presplit_host_reclaim()
                eo.presplit_host_reclaim()
        end = [m for m in logs.output if "BOOTZEIT5 LOAD-GC-FREEZE end" in m][0]
        self.assertRegex(end, r"reclaim gc=[\d.]+ s over 2 layers \([\d.]+ s/layer\)")
        self.assertRegex(end, r"nonreclaim\(anon\+shmem\) start=\S+ peak=\S+ GiB")

    def test_an_exception_in_the_load_still_unfreezes(self):
        with envs.SGLANG_OPT_LOAD_GC_FREEZE.override(True):
            with self.assertRaises(RuntimeError):
                with load_gc_frozen(what="test", expert_presplit=True):
                    raise RuntimeError("load died")
        self.assertLess(gc.get_freeze_count(), load_gc._FOREIGN_FREEZE_MIN)

    def test_someone_elses_freeze_is_left_alone(self):
        gc.freeze()
        theirs = gc.get_freeze_count()
        with envs.SGLANG_OPT_LOAD_GC_FREEZE.override(True):
            with load_gc_frozen(what="test", expert_presplit=True) as frozen:
                self.assertIsNone(frozen)
        self.assertEqual(gc.get_freeze_count(), theirs)

    def test_switch_default_on_and_off_freezes_nothing(self):
        self.assertTrue(envs.SGLANG_OPT_LOAD_GC_FREEZE.get())
        with envs.SGLANG_OPT_LOAD_GC_FREEZE.override(False):
            with load_gc_frozen(what="test", expert_presplit=True) as frozen:
                self.assertIsNone(frozen)
                self.assertLess(gc.get_freeze_count(), load_gc._FOREIGN_FREEZE_MIN)


class TestCgroupReader(unittest.TestCase):
    def test_nonreclaim_is_anon_plus_shmem_and_none_when_absent(self):
        # bootzeit_eval's HOST-LADESPITZE currency; a missing term is None,
        # never a partial sum.
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "memory.stat")
            with open(p, "w") as f:
                f.write("anon 1000\nfile 5000\nshmem 300\nfile_mapped 7\n")
            self.assertEqual(load_gc.cgroup_nonreclaim_bytes(p), 1300)
            with open(p, "w") as f:
                f.write("anon 1000\nfile 5000\n")
            self.assertIsNone(load_gc.cgroup_nonreclaim_bytes(p))
        self.assertIsNone(load_gc.cgroup_nonreclaim_bytes("/nonexistent/memory.stat"))


class TestDefaultLoaderCallEdge(unittest.TestCase):
    def test_model_build_and_weight_load_run_inside_the_gated_freeze(self):
        # Both the construction (so the model's objects are born after the
        # freeze) and the weight load (where the presplit collects) sit inside
        # `with load_gc_frozen(..., expert_presplit=expert_presplit_runs())`.
        src = textwrap.dedent(inspect.getsource(loader_mod.DefaultModelLoader.load_model))
        fn = ast.parse(src).body[0]
        inside, gate = set(), []
        for node in ast.walk(fn):
            if not isinstance(node, ast.With):
                continue
            calls = [
                i.context_expr for i in node.items
                if isinstance(i.context_expr, ast.Call)
                and isinstance(i.context_expr.func, ast.Name)
                and i.context_expr.func.id == "load_gc_frozen"
            ]
            if not calls:
                continue
            for kw in calls[0].keywords:
                if kw.arg == "expert_presplit" and isinstance(kw.value, ast.Call):
                    gate.append(kw.value.func.id)
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    f = sub.func
                    inside.add(f.id if isinstance(f, ast.Name) else getattr(f, "attr", ""))
        self.assertEqual(gate, ["expert_presplit_runs"])
        self.assertIn("_initialize_model", inside)
        self.assertIn("load_weights_and_postprocess", inside)


if __name__ == "__main__":
    unittest.main()
