"""y7u (02.10.): SchedulerWeightUpdaterManager is a ``slots=True`` dataclass,
so every ``self.<name> = ...`` in its methods must name a declared field.

y7u died on all six ranks in its first flip on ``AttributeError:
'SchedulerWeightUpdaterManager' object has no attribute
'_pdflip_leg_order_na_epoch'`` -- and y7t had silently logged 0 PDFLIP-LEG-ORDER
lines because the same mistake one line earlier was swallowed.  This is the
fourth time the lesson in the class comments was learnt at the metal, so the
check is static and covers the whole class.
"""

import ast
import pathlib
import unittest

_SRC = (
    pathlib.Path(__file__).resolve().parents[4]
    / "python/flliper/srt/managers/scheduler_components/weight_updater.py"
)


def _undeclared_assignments():
    tree = ast.parse(_SRC.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "SchedulerWeightUpdaterManager"
    )
    declared = {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)}
    declared |= {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    bad = {}
    for node in ast.walk(cls):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Store)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr not in declared
        ):
            bad.setdefault(node.attr, []).append(node.lineno)
    return bad


class TestWeightUpdaterSlotsFields(unittest.TestCase):
    def test_every_self_assignment_is_a_declared_field(self):
        self.assertEqual(_undeclared_assignments(), {})

    def test_leg_order_epochs_are_fields(self):
        from flliper.srt.managers.scheduler_components.weight_updater import (
            SchedulerWeightUpdaterManager,
        )

        names = SchedulerWeightUpdaterManager.__slots__
        self.assertIn("_pdflip_leg_order_epoch", names)
        self.assertIn("_pdflip_leg_order_na_epoch", names)


if __name__ == "__main__":
    unittest.main()
