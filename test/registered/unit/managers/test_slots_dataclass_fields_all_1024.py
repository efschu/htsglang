"""27B-RC41-1024 (Qwen review qwen-review-27b-rc41-nachzuege-1024 point 1): the y7u guard, extended from ONE class to
EVERY ``slots=True`` / ``frozen=True`` dataclass and every explicit-``__slots__`` class under ``python/``.

The slots lesson was learnt at the metal five times in ``SchedulerWeightUpdaterManager`` alone (y7u, y4j-po, fnFL2x5/x7,
#108, 27b-drei-bootflip-1024 = 4fbdd3e9d2: ``_pdflip_wake_defer_work`` killed every 27B image from rc37 at the first
flip), and test_weight_updater_slots_fields_y7u.py guards only that class.  The repo-wide census for rc41 found the same
defect class in ``SchedulerBatchResultProcessor`` (``@dataclass(kw_only=True, slots=True, frozen=True)``):
``self._997_seen`` / ``_997_unreadable`` / ``_997_emitted`` were assigned undeclared inside ``try/except Exception:
pass`` -- no rank death, but the #997b OUTPUT-FILL instrument raised on its first write and never logged a line.

Static, like y7u: an AST walk, no imports of the scanned modules.  What counts as a defect:
  * a ``slots=True`` dataclass whose method stores ``self.<name>`` (or ``setattr(self, "<name>", ...)``) for a name that
    is neither a field of the class nor of a slots base (a base without slots, or one this walk cannot resolve, gives
    the instance a ``__dict__`` and ends the check for that class);
  * a ``frozen=True`` dataclass whose method stores ``self.<name>`` at all (``FrozenInstanceError``; the frozen idiom is
    ``object.__setattr__``, which is not an attribute store);
  * an explicit-``__slots__`` class whose method stores ``self.<name>`` outside its own and its bases' ``__slots__``.
Exempt by construction (not by a list): ``setattr(self, name, v)`` with a NON-constant name -- ``SplitCounters``
(decode_host_split.py) and ``HostCostCounters`` (host_round_cost.py) zero their counters with
``for name in self.__slots__: setattr(self, name, 0.0)``, which can only name declared slots.
"""
from __future__ import annotations

import ast
import pathlib
import unittest
import warnings
from typing import Dict, List, Optional, Set, Tuple

_ROOT = pathlib.Path(__file__).resolve().parents[4] / "python"

_OPEN_BASES = {"object", "Generic", "Protocol", "ABC"}


def _dataclass_kw(cls: ast.ClassDef) -> Optional[Dict[str, object]]:
    for d in cls.decorator_list:
        f = d.func if isinstance(d, ast.Call) else d
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if name == "dataclass":
            if not isinstance(d, ast.Call):
                return {}
            return {k.arg: (k.value.value if isinstance(k.value, ast.Constant) else None) for k in d.keywords}
    return None


def _explicit_slots(cls: ast.ClassDef) -> Optional[Set[str]]:
    for n in cls.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__slots__" for t in n.targets):
            try:
                v = ast.literal_eval(n.value)
            except Exception:  # noqa: BLE001 - a computed __slots__ is not checkable here
                return None
            return {v} if isinstance(v, str) else set(v)
    return None


def _setters(cls: ast.ClassDef) -> Set[str]:
    out = set()
    for n in cls.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in n.decorator_list:
                if isinstance(d, ast.Attribute) and d.attr == "setter":
                    out.add(n.name)
    return out


class _Index:
    def __init__(self, sources: Dict[str, str]):
        self.classes: Dict[str, List[Tuple[str, ast.ClassDef]]] = {}
        for path, text in sources.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)  # invalid escapes in scanned modules are not ours to judge
                tree = ast.parse(text)
            for n in ast.walk(tree):
                if isinstance(n, ast.ClassDef):
                    self.classes.setdefault(n.name, []).append((path, n))

    def declared(self, cls: ast.ClassDef, seen: Optional[Set[int]] = None) -> Optional[Set[str]]:
        """Names an instance of ``cls`` may store, or None when it has a ``__dict__`` (or this walk cannot tell)."""
        seen = set() if seen is None else seen
        if id(cls) in seen:
            return set()
        seen.add(id(cls))
        kw = _dataclass_kw(cls)
        own = _explicit_slots(cls)
        if own is None and not (kw and kw.get("slots")):
            return None
        names = set(own or ())
        if own is not None and "__dict__" in names:
            return None
        if kw is not None:
            for n in cls.body:
                if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                    if "ClassVar" not in ast.unparse(n.annotation):
                        names.add(n.target.id)
        names |= _setters(cls)
        for b in cls.bases:
            bn = b.attr if isinstance(b, ast.Attribute) else getattr(b, "id", None)
            if isinstance(b, ast.Subscript):
                bn = getattr(b.value, "id", getattr(b.value, "attr", None))
            if bn in _OPEN_BASES:
                continue
            cands = self.classes.get(bn or "")
            if not cands or len(cands) > 1:
                return None  # unknown or ambiguous base: assume a __dict__, never a false red
            sub = self.declared(cands[0][1], seen)
            if sub is None:
                return None
            names |= sub
        return names


def _methods(cls: ast.ClassDef):
    for fn in cls.body:
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or not fn.args.args:
            continue
        if fn.name == "__new__" or any(
            isinstance(d, ast.Name) and d.id in ("staticmethod", "classmethod") for d in fn.decorator_list
        ):
            continue
        yield fn, fn.args.args[0].arg


def find_defects(sources: Dict[str, str]) -> Dict[str, List[str]]:
    """``{"<path>::<Class>": ["<kind> <name> :<line> in <method>", ...]}`` for every defect described above."""
    idx = _Index(sources)
    bad: Dict[str, List[str]] = {}
    for name, lst in idx.classes.items():
        for path, cls in lst:
            kw = _dataclass_kw(cls)
            frozen = bool(kw and kw.get("frozen"))
            decl = idx.declared(cls)
            if decl is None and not frozen:
                continue
            setters = _setters(cls)
            for fn, me in _methods(cls):
                for node in ast.walk(fn):
                    stored = None
                    if (
                        isinstance(node, ast.Attribute)
                        and isinstance(node.ctx, (ast.Store, ast.Del))
                        and isinstance(node.value, ast.Name)
                        and node.value.id == me
                    ):
                        stored = node.attr
                        if frozen and stored not in setters:
                            bad.setdefault(f"{path}::{name}", []).append(f"FROZEN-STORE {stored} :{node.lineno} in {fn.name}")
                    elif (
                        isinstance(node, ast.Call)
                        and ast.unparse(node.func) in ("setattr", "object.__setattr__")
                        and len(node.args) >= 2
                        and isinstance(node.args[0], ast.Name)
                        and node.args[0].id == me
                        and isinstance(node.args[1], ast.Constant)
                    ):
                        stored = node.args[1].value
                    if stored is not None and decl is not None and stored not in decl:
                        bad.setdefault(f"{path}::{name}", []).append(f"UNDECLARED {stored} :{node.lineno} in {fn.name}")
    return bad


def _tree_sources() -> Dict[str, str]:
    out = {}
    for p in sorted(_ROOT.rglob("*.py")):
        rel = str(p.relative_to(_ROOT))
        if "/tests/fixtures/" in f"/{rel}":
            continue
        out[rel] = p.read_text(encoding="utf-8-sig", errors="replace")
    return out


class TestSlotsDataclassFieldsAll(unittest.TestCase):
    def test_the_checker_sees_the_three_shapes(self):
        src = (
            "from dataclasses import dataclass, field\n"
            "@dataclass(kw_only=True, slots=True)\n"
            "class M:\n"
            "    a: int = 0\n"
            "    def f(self):\n"
            "        self.a = 1\n"
            "        self._late = 2\n"
            "        setattr(self, '_late2', 3)\n"
            "        for n in ('a',):\n"
            "            setattr(self, n, 0)\n"
            "@dataclass(frozen=True, slots=True)\n"
            "class F:\n"
            "    a: int = 0\n"
            "    def __post_init__(self):\n"
            "        object.__setattr__(self, 'a', 1)\n"
            "    def g(self):\n"
            "        self.a = 2\n"
            "class S:\n"
            "    __slots__ = ('x',)\n"
            "    def __init__(self):\n"
            "        self.x = 0\n"
            "        self.y = 1\n"
            "class Open:\n"
            "    def h(self):\n"
            "        self.anything = 1\n"
            "@dataclass(slots=True)\n"
            "class OnOpenBase(Open):\n"
            "    def k(self):\n"
            "        self.free = 1\n"
        )
        bad = find_defects({"m.py": src})
        self.assertEqual(
            bad,
            {
                "m.py::M": ["UNDECLARED _late :7 in f", "UNDECLARED _late2 :8 in f"],
                "m.py::F": ["FROZEN-STORE a :17 in g"],
                "m.py::S": ["UNDECLARED y :22 in __init__"],
            },
        )

    def test_the_rc40_defect_is_seen_in_its_real_shape(self):
        # the exact store 4fbdd3e9d2 fixed, against the real class declaration shape
        src = (
            "from dataclasses import dataclass\n"
            "@dataclass(kw_only=True, slots=True)\n"
            "class SchedulerWeightUpdaterManager:\n"
            "    pdflip_fence_raised: bool = False\n"
            "    def _pdflip_kv_clear_part(self, _rw_work):\n"
            "        self._pdflip_wake_defer_work = bool(_rw_work)\n"
        )
        self.assertIn("weight_updater.py::SchedulerWeightUpdaterManager", find_defects({"weight_updater.py": src}))

    def test_the_997b_counters_are_a_declared_slot(self):
        import os

        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        from dataclasses import fields

        from flliper.srt.managers.scheduler_components.batch_result_processor import (
            SchedulerBatchResultProcessor,
        )

        self.assertIn("_997_counts", SchedulerBatchResultProcessor.__slots__)
        f = {x.name: x for x in fields(SchedulerBatchResultProcessor)}["_997_counts"]
        self.assertFalse(f.init)
        self.assertFalse(f.compare)
        self.assertEqual(f.default_factory(), {})

    def test_no_slots_or_frozen_class_stores_an_undeclared_attribute(self):
        self.assertEqual(find_defects(_tree_sources()), {})


if __name__ == "__main__":
    unittest.main()
