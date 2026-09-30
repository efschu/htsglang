"""Every ``self.<name> = ...`` in ``SchedulerWeightUpdaterManager`` hits a declared slot.

The class is a ``slots=True`` dataclass, so an attribute written without a field
raises AttributeError at the write -- and only on the path that writes it. Four
times a new attribute reached the metal that way: 7a3d8f5ceb
(``_weg2_last_inject_cover``, fnFL2x5/x7), #1378 (``_weg2_owned_name_keys_cache``),
fnFL2x82 (``_weg2_xchg_leg_lock``) and 0a78051a4d (``_weg2_resident_prefetch``,
y4j-po 30.09.: all three D ranks at the first sleep with PAUSE-OVERLAP on). The
per-field tests only guard the field they were written for; this one guards the
class, by reading every assignment target in its source.
"""

import ast
import pathlib
import unittest

from sglang.srt.managers.scheduler_components import weight_updater as wu

SRC = (
    pathlib.Path(__file__).resolve().parents[4]
    / "python/sglang/srt/managers/scheduler_components/weight_updater.py"
)
CLS = wu.SchedulerWeightUpdaterManager


def _slots(cls) -> set:
    out = set()
    for k in cls.__mro__:
        out |= set(getattr(k, "__slots__", ()) or ())
    return out


def _self_writes(class_node: ast.ClassDef) -> dict:
    """{attr: [lineno, ...]} of every ``self.attr`` assignment target in the class."""
    out: dict = {}
    for node in ast.walk(class_node):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        else:
            continue
        for t in targets:
            for sub in ast.walk(t):
                if (isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name)
                        and sub.value.id == "self"):
                    out.setdefault(sub.attr, []).append(sub.lineno)
    return out


class TestWeightUpdaterSlotsGuard(unittest.TestCase):
    def _class_node(self) -> ast.ClassDef:
        tree = ast.parse(SRC.read_text())
        return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == CLS.__name__)

    def test_the_class_is_slotted(self):
        # the guard below only means something while the class refuses ad-hoc attributes
        self.assertTrue(hasattr(CLS, "__slots__"))
        with self.assertRaises(AttributeError):
            CLS.__new__(CLS).__dict__  # noqa: B018

    def test_every_self_write_is_a_declared_slot(self):
        writes = _self_writes(self._class_node())
        self.assertTrue(writes, "the scan found no self writes -- the parser missed the class")
        missing = {a: lines for a, lines in writes.items() if a not in _slots(CLS)}
        self.assertEqual(missing, {}, "self.<attr> written without a field on the slots=True "
                         "dataclass (AttributeError at the write): %r" % missing)

    def test_resident_prefetch_is_a_field(self):
        # y4j-po: the PAUSE-OVERLAP scope writes it before the sleep loop
        inst = CLS.__new__(CLS)
        inst._weg2_resident_prefetch = {"weights_0": 1}
        self.assertEqual(inst._weg2_resident_prefetch, {"weights_0": 1})
        self.assertIsNone(CLS.__dataclass_fields__["_weg2_resident_prefetch"].default)


if __name__ == "__main__":
    unittest.main()
