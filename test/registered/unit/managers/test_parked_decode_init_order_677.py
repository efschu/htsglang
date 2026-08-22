# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#677: the park set must be built AFTER the pool it reads.

THE DEFECT THIS EXISTS FOR, and why nothing else could catch it
---------------------------------------------------------------
Metal, 2026-08-16 09:00:45, all three ranks dead at construction::

    line 655,  in __init__            -> self.init_model_worker()
    line 1420, in init_model_worker   -> self.init_parked_decode_set()
    line 2840, in init_parked_decode_set
        mamba_allocator = getattr(self.req_to_token_pool, "mamba_allocator")
    AttributeError: 'Scheduler' object has no attribute 'req_to_token_pool'

The attribute name was RIGHT and the attribute is NOT TP-stack-only -- the PP
scheduler has it. What was wrong was WHEN. ``init_admission_limiter`` lives
inside ``init_model_worker``, the new call was placed next to it because both
concern the concurrency cap, and ``self.req_to_token_pool`` is assigned in
``__init__`` from the RESULT of ``init_model_worker()``, four lines after it
returns. So the park set asked the memory pool for its slot count while the
memory pool was still being built.

WHY THE OTHER 31 TESTS PASSED. They bind the methods to a stand-in that
already carries the fields, which is the honest way to test the ARITHMETIC and
is structurally incapable of testing the ORDER. A constructed Scheduler is the
only thing that shows it, and a Scheduler cannot be constructed on CPU: it
needs a model, a device, a process group and a loaded checkpoint.

So this test reads the SOURCE. That is a deliberate choice, not a shortcut: the
invariant is a static property of the constructor -- "the call comes after the
assignment" -- and a static property is exactly what a source-level assertion
can hold. It fails if anyone moves the call back inside ``init_model_worker``,
which is the specific regression that cost three boots.
"""

import ast
import inspect
import unittest

from sglang.srt.managers import scheduler as scheduler_mod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)

POOL_ATTR = "req_to_token_pool"
CALL = "init_parked_decode_set"


def _class_node(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _method(cls_node, name):
    for node in cls_node.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return node
    raise AssertionError(f"method {name} not found on {cls_node.name}")


def _self_calls(node, name):
    """Line numbers of `self.<name>()` calls inside node."""
    out = []
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == name
            and isinstance(sub.func.value, ast.Name)
            and sub.func.value.id == "self"
        ):
            out.append(sub.lineno)
    return out


def _self_assign_lines(node, attr):
    """Line numbers where `self.<attr> = ...` is assigned inside node."""
    out = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Assign):
            continue
        for tgt in sub.targets:
            if (
                isinstance(tgt, ast.Attribute)
                and tgt.attr == attr
                and isinstance(tgt.value, ast.Name)
                and tgt.value.id == "self"
            ):
                out.append(sub.lineno)
    return out


class TheParkSetIsBuiltAfterThePoolItReads(unittest.TestCase):
    def setUp(self):
        src = inspect.getsource(scheduler_mod)
        self.tree = ast.parse(src)
        self.sched = _class_node(self.tree, "Scheduler")
        self.init = _method(self.sched, "__init__")

    def test_the_call_happens_in_init_not_in_init_model_worker(self):
        """The exact regression: placed by topic instead of by dependency."""
        imw = _method(self.sched, "init_model_worker")
        self.assertEqual(
            [],
            _self_calls(imw, CALL),
            f"self.{CALL}() is called inside init_model_worker, which runs "
            f"BEFORE self.{POOL_ATTR} is assigned from its return value. "
            "This is the 2026-08-16 09:00:45 AttributeError that killed all "
            "three ranks at construction.",
        )
        self.assertEqual(
            1,
            len(_self_calls(self.init, CALL)),
            f"self.{CALL}() must be called exactly once, from __init__.",
        )

    def test_the_call_comes_after_the_pool_assignment(self):
        call_lines = _self_calls(self.init, CALL)
        assign_lines = _self_assign_lines(self.init, POOL_ATTR)
        self.assertTrue(
            assign_lines,
            f"self.{POOL_ATTR} is no longer assigned in __init__; this guard "
            "is now watching the wrong thing and must be re-aimed rather than "
            "deleted.",
        )
        self.assertTrue(call_lines, f"self.{CALL}() is not called from __init__")
        self.assertGreater(
            min(call_lines),
            max(assign_lines),
            f"self.{CALL}() at line {min(call_lines)} runs before "
            f"self.{POOL_ATTR} is assigned at line {max(assign_lines)}. It "
            "reads that pool for the GDN slot count, so it would raise "
            "AttributeError at construction on every rank.",
        )

    def test_the_body_still_reads_the_attribute_this_guard_orders(self):
        """If the read moves, this whole guard is guarding nothing.

        Stated as its own failure rather than folded into the others, so a
        future refactor that renames the input gets told the guard is stale
        instead of silently keeping a green test that proves nothing.
        """
        body = _method(self.sched, CALL)
        reads = [
            sub.attr
            for sub in ast.walk(body)
            if isinstance(sub, ast.Attribute)
            and isinstance(sub.value, ast.Name)
            and sub.value.id == "self"
        ]
        self.assertIn(
            POOL_ATTR,
            reads,
            f"{CALL} no longer reads self.{POOL_ATTR}; re-aim this ordering "
            "guard at whatever input it reads now.",
        )


if __name__ == "__main__":
    unittest.main()
