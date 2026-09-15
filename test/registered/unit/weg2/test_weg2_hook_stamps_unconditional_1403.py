"""#1403 (boot xsn135, 2026-09-15): the exchange hook's phase stamps must
exist on every boot, not only under the lane-coverage arm.

The first boot without --xchg-coverage-diff raised UnboundLocalError on the
first `_hk_ph(...)` in `_weg2_shadow_hook`, the observer-except reported "the
flip is unaffected", P never deposited, and D's wake waited for the front's
W4. This test parses the function and refuses a `_hk_ph` definition that
sits under an `if` again.
"""

import ast
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sglang.srt.managers.scheduler_components.weight_updater as wu

SRC = wu.__file__.replace(".pyc", ".py")


def _hook_fn(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_weg2_shadow_hook":
            return node
    raise AssertionError("_weg2_shadow_hook not found")


def test_hk_ph_is_defined_at_function_level_not_under_an_if():
    with open(SRC, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    fn = _hook_fn(tree)
    parents = {}
    for parent in ast.walk(fn):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    defs = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.FunctionDef) and n.name == "_hk_ph"
    ]
    assert defs, "the hook has lost its _hk_ph stamp function"
    for d in defs:
        p = parents.get(d)
        chain = []
        while p is not None and p is not fn:
            chain.append(type(p).__name__)
            p = parents.get(p)
        assert "If" not in chain, (
            "_hk_ph is defined under an `if` (%s): a boot that does not take "
            "that branch raises UnboundLocalError at the first stamp (xsn135)"
            % " > ".join(chain)
        )


def test_every_stamp_name_is_used_after_a_definition():
    with open(SRC, encoding="utf-8") as f:
        lines = f.read().split("\n")
    first_def = next(i for i, l in enumerate(lines) if "def _hk_ph(name):" in l)
    first_use = next(i for i, l in enumerate(lines) if "_hk_ph(\"" in l or "_hk_ph('" in l)
    assert first_def < first_use
