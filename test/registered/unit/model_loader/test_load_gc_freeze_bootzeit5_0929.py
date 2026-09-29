# SPDX-License-Identifier: Apache-2.0
"""BOOTZEIT 5 (29.09.): the model load runs with the pre-load objects frozen.

Metal (z30w-park): the presplit's "repack" posten is a FIXED cost per MoE
layer -- 0.43 s on PP0 (512 of 512 rows repacked), 0.49 s on D TP0 (29 of 201
rows), Marlin itself 0.01 s -- and the fixed piece in it is the per-layer full
``gc.collect()`` over the whole process heap (0.26-0.29 s measured on the
import graph alone, 801522 objects; 1.5 ms after ``gc.freeze()``).

What the freeze must never change: objects the load creates stay collectible,
cycles included, and the process leaves the load with the permanent
generation as it found it.
"""

import ast
import gc
import inspect
import textwrap
import unittest
import weakref

from sglang.srt.environ import envs
from sglang.srt.model_loader import loader as loader_mod
from sglang.srt.model_loader.load_gc import load_gc_frozen


class _Node:
    pass


def _garbage_cycle():
    """A reference cycle only a collection can free; returns a weakref to it."""
    a, b = _Node(), _Node()
    a.other, b.other = b, a
    a.payload = bytearray(1 << 20)
    return weakref.ref(a)


class TestLoadGcFrozen(unittest.TestCase):
    def setUp(self):
        gc.unfreeze()  # a clean permanent generation for every case
        gc.disable()  # only the explicit collections below run (deterministic)

    def tearDown(self):
        gc.enable()
        gc.unfreeze()

    def test_cycles_the_load_creates_are_still_collected(self):
        # Derived property: the freeze exempts what existed BEFORE; a cycle
        # born inside (the [E] stacks' case) is found by the per-layer collect.
        with envs.SGLANG_OPT_LOAD_GC_FREEZE.override(True):
            with load_gc_frozen(what="test") as frozen:
                self.assertGreater(frozen, 0)
                self.assertGreater(gc.get_freeze_count(), 0)
                ref = _garbage_cycle()
                self.assertIsNotNone(ref())
                self.assertGreaterEqual(gc.collect(), 2)
                self.assertIsNone(ref())
        self.assertEqual(gc.get_freeze_count(), 0)

    def test_a_cycle_from_before_the_load_waits_until_after_it(self):
        # The other half of the contract: pre-load objects are not walked
        # during the load (that walk is the cost), and nothing is lost -- the
        # unfreeze hands them back and the next collection frees them.
        ref = _garbage_cycle()
        with envs.SGLANG_OPT_LOAD_GC_FREEZE.override(True):
            with load_gc_frozen(what="test"):
                gc.collect()
                self.assertIsNotNone(ref())
        gc.collect()
        self.assertIsNone(ref())

    def test_an_exception_in_the_load_still_unfreezes(self):
        with envs.SGLANG_OPT_LOAD_GC_FREEZE.override(True):
            with self.assertRaises(RuntimeError):
                with load_gc_frozen(what="test"):
                    raise RuntimeError("load died")
        self.assertEqual(gc.get_freeze_count(), 0)

    def test_someone_elses_freeze_is_left_alone(self):
        # Negative branch: gc.unfreeze() cannot release only our part, so a
        # populated permanent generation means hands off, in and out.
        gc.freeze()
        theirs = gc.get_freeze_count()
        with envs.SGLANG_OPT_LOAD_GC_FREEZE.override(True):
            with load_gc_frozen(what="test") as frozen:
                self.assertIsNone(frozen)
        self.assertEqual(gc.get_freeze_count(), theirs)

    def test_default_on_and_off_freezes_nothing(self):
        self.assertTrue(envs.SGLANG_OPT_LOAD_GC_FREEZE.get())
        with envs.SGLANG_OPT_LOAD_GC_FREEZE.override(False):
            with load_gc_frozen(what="test") as frozen:
                self.assertIsNone(frozen)
                self.assertEqual(gc.get_freeze_count(), 0)


class TestDefaultLoaderCallEdge(unittest.TestCase):
    def test_model_build_and_weight_load_run_inside_the_freeze(self):
        # Call edge: both the construction (so the model's objects are born
        # after the freeze) and the weight load (where the presplit collects)
        # sit inside the `with load_gc_frozen(...)` of DefaultModelLoader.
        src = textwrap.dedent(inspect.getsource(loader_mod.DefaultModelLoader.load_model))
        fn = ast.parse(src).body[0]
        inside = set()
        for node in ast.walk(fn):
            if not isinstance(node, ast.With):
                continue
            ctx = [
                i.context_expr.func.id
                for i in node.items
                if isinstance(i.context_expr, ast.Call)
                and isinstance(i.context_expr.func, ast.Name)
            ]
            if "load_gc_frozen" not in ctx:
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    f = sub.func
                    inside.add(f.id if isinstance(f, ast.Name) else getattr(f, "attr", ""))
        self.assertIn("_initialize_model", inside)
        self.assertIn("load_weights_and_postprocess", inside)


if __name__ == "__main__":
    unittest.main()
