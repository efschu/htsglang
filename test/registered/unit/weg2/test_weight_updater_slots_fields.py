"""SchedulerWeightUpdaterManager is a ``slots=True`` dataclass: every attribute a
method WRITES must be a declared field, or the write raises AttributeError at
runtime -- in rc8a on the first wake of every rank (``_weg2_nvfp4_draft_disk_reloaded``).
This is the sixth instance of the class, so the check is structural (AST), not
per attribute."""

import ast
import pathlib
import unittest

SRC = (
    pathlib.Path(__file__).resolve().parents[4]
    / "python/sglang/srt/managers/scheduler_components/weight_updater.py"
)
CLS = "SchedulerWeightUpdaterManager"


def _class_node(tree):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == CLS:
            return node
    raise AssertionError(f"{CLS} not found in {SRC}")


def _declared(cls):
    names = set()
    for stmt in cls.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            names.add(stmt.target.id)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(stmt.name)
    return names


def _self_writes(cls):
    out = {}
    for fn in cls.body:
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            for t in targets:
                for sub in ast.walk(t):
                    if (
                        isinstance(sub, ast.Attribute)
                        and isinstance(sub.value, ast.Name)
                        and sub.value.id == "self"
                        and isinstance(sub.ctx, ast.Store)
                    ):
                        out.setdefault(sub.attr, node.lineno)
    return out


class TestSlotsFields(unittest.TestCase):
    def test_every_self_write_is_a_declared_field(self):
        cls = _class_node(ast.parse(SRC.read_text()))
        declared = _declared(cls)
        missing = {a: ln for a, ln in _self_writes(cls).items() if a not in declared}
        self.assertEqual(
            missing,
            {},
            "slots=True dataclass: these self.<attr> writes have no field "
            "(weight_updater.py line of first write) -> AttributeError at runtime",
        )

    def test_the_class_really_is_slotted(self):
        from sglang.srt.managers.scheduler_components import weight_updater as wu

        self.assertTrue(hasattr(getattr(wu, CLS), "__slots__"))


if __name__ == "__main__":
    unittest.main()
